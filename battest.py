from ctypes import ArgumentError
from turtle import back
from typing import Any

from siglent.siglent.multimeters import SDM3000X, SDMMeasurement, SDMVoltageRange
from siglent.siglent.power_supplies import SPD4000X, RunMode
from siglent.siglent.dc_loads import SDL1000X, LoadFunction

from pyvisa import ResourceManager

import PySimpleGUI as sg

from dataclasses import dataclass, asdict

from typed_configparser import ConfigParser

from statistics import mean

from enum import StrEnum

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import logging
import time

WINDOW_LOCATION = (200, 200)

class TestType(StrEnum):
    TEST_FULL = "Full"
    TEST_QUICK = "Quick"

@dataclass
class Config:
    psuIP: str = "192.168.1.35"
    psuCh: int = 2
    dclIP: str = "192.168.1.34"
    dmmIP: str = "192.168.1.32"
    chargeC: float = 1.0        # Charge current in amps
    chargeV: float = 3.6        # Voltage below which the battery will be charged before the test
    dischargeC: float = 1.0     # Discharge current in amps
    dischargeV: float = 3.8     # Discharge cutoff voltage in volts
    dischargeT: int = 120       # Charge/discharge timeout in seconds
    dcrR: float = 0.2           # Max DC resistance of battery for pass/fail
    testLeadR: float = 0.02     # Test lead resistance between multimeter connection and battery

mainLayout = [
    # Header
    [
        sg.Text("SWL Node Battery Tester")
    ],
    # Instrument IPs
    [
        sg.Column([
            [sg.Text("Power Supply IP")],
            [sg.Text("Power Supply Channel")],
            [sg.Text("DC Load IP")],
            [sg.Text("Multimeter IP")],
        ]),
        sg.Column([
            [sg.In(size=(32,1), enable_events=True, key="PSU_IP")],
            [sg.Spin(list(range(1,5)), 1, size = (5,1), key="PSU_CH")],
            [sg.In(size=(32,1), enable_events=True, key="DCL_IP")],
            [sg.In(size=(32,1), enable_events=True, key="DMM_IP")]
        ])
    ],
    # Test Config
    [
        sg.Column([
            [sg.Text("Charge Current (A)")],
            [sg.Text("Charge Voltage Threshold (V)")],
            [sg.Text("Discharge Current (A)")],
            [sg.Text("Discharge Voltage Threshold (V)")],
            [sg.Text("Charge/Discharge Timeout (s)")],
            [sg.Text("Battery Max DCR Threshold (Ω)")],
            [sg.Text("Test Lead Resistance (Ω)")]
        ]),
        sg.Column([
            [sg.Spin(np.arange(0.1,2.0,0.05).tolist(), 1.0, size = (5,1), key="CHG_A")],
            [sg.Spin(np.arange(3.0,3.7,0.1).tolist(), 3.6, size = (5,1), key="CHG_V")],
            [sg.Spin(np.arange(0.1,2.0,0.05).tolist(), 1.0, size = (5,1), key="DSG_A")],
            [sg.Spin(np.arange(3.7,4.2,0.1).tolist(), 3.8, size = (5,1), key="DSG_V")],
            [sg.Spin(np.arange(10,3600,1).tolist(), 60, size = (5,1), key="DSG_T")],
            [sg.Spin(np.arange(0.0,1.0,0.05).tolist(), 0.20, size = (5,1), key="DCR_R")],
            [sg.Spin(np.arange(0.0,1.0,0.001).tolist(), 0.020, size = (5,1), key="TEST_R")]
        ])
    ],
    # Canvas for plots
    [
        sg.Canvas(key="PLOT")
    ],
    # Bottom Butons
    [
        sg.Button("Start", key="BTN_START"),
        sg.Button("Save Config", key="BTN_SAVECFG"),
        sg.Button("Exit", key="BTN_EXIT"),
        sg.Text("Initialized, idle", key="STATUS_TEXT")
    ]
]

mainWindow = sg.Window(title="SWL Node Battery Tester", layout=mainLayout, finalize=True, location=WINDOW_LOCATION)

configPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("pyvisa").setLevel(logging.INFO)

# Test tracking variables
testRunning : bool = False
stopTest : bool = False

# Global instruments
psu: SPD4000X
dcl: SDL1000X
dmm: SDM3000X

# Data storage for battery measurements, each measurement is in format [timestamp, value]
psu_v : list[tuple[int,float]] = []
psu_c : list[tuple[int,float]] = []
dcl_v : list[tuple[int,float]] = []
dcl_c : list[tuple[int,float]] = []
dmm_v : list[tuple[int,float]] = []

def time_ms() -> int:
    return int(time.time_ns() // 1_000_000)

def charge(config: Config, target_v: float, timeout_s: int = 3600) -> bool:
    """
    Charge the battery using the given config and instruments

    Args:
        config (Config): _description_
        psu (SPD4000X): _description_
        dmm (SDM3000X): _description_

    Returns:
        bool: True on success, False on failure
    """
    # Globals
    global psu, dcl, dmm
    global psu_v, psu_c, dcl_v, dcl_c, dmm_v
    global stopTest

    # Print
    logger.info(f"Preparing for battery charge cycle. C = {config.chargeC} A, V = {target_v}")

    # Clear instruments
    psu.clear()
    dmm.clear()
    
    # Configure PSU for charge profile
    psu.channel(config.psuCh).voltage = target_v
    psu.channel(config.psuCh).current = config.chargeC
    psu.channel(config.psuCh).ovp = target_v * 1.25
    psu.channel(config.psuCh).ocp = config.chargeC * 1.25
    
    # Configure DMM for measurement of battery voltage
    dmm.measurement = SDMMeasurement.DCV
    dmm.dcv_range = SDMVoltageRange.V_20V
    
    # Wait for instruments
    psu.block_until_complete()
    dmm.block_until_complete()
    
    # Enable power supply
    psu.channel(config.psuCh).output = True
    time.sleep(1)
    
    # Start time
    start = time.time()
    # Flag to break out of loop
    charged: bool = False
    
    # Loop while we're not charged and haven't timed out
    while not charged and time.time() < (start + timeout_s) and not stopTest:
        # Measure instrument readings
        psu_c.append((time_ms(), psu.channel(config.psuCh).meas_current))
        psu_v.append((time_ms(), psu.channel(config.psuCh).meas_voltage))
        dmm_v.append((time_ms(), dmm.value))
        # Calculate new averages
        psu_c_avg = mean([item[1] for item in psu_c[-5:]])
        psu_v_avg = mean([item[1] for item in psu_v[-5:]])
        dmm_v_avg = mean([item[1] for item in dmm_v[-5:]])
        # Print
        logger.info(f"  - Charge Current: {psu_c_avg:.3f} A, Charge Voltage: {dmm_v_avg:.3f} V")
        # Check if we hit our charge current cutoff (0.02 * C) and we have taken at least 5 samples
        if psu_c_avg <= (0.02 * config.chargeC) and len(psu_c) > 5:
            logger.info("  - Hit charging current cutoff, fully charged!")
            charged = True
            break
        # Delay
        time.sleep(0.5)

    # Shut off PSU
    psu.channel(config.psuCh).output = False

    if not charged:
        logger.error("Timed out before charging completed!")
        return False
    else:
        logger.info("Charging complete!")
        return True

def discharge(config: Config, target_v: float, timeout_s = 180) -> bool:
    """
    Discharge the battery to the specified target voltage

    Args:
        config (Config): _description_
        dcl (SDL1000X): _description_
        dmm (SDM3000X): _description_

    Returns:
        bool: True on success, False on failure
    """
    # Globals
    global psu, dcl, dmm
    global psu_v, psu_c, dcl_v, dcl_c, dmm_v
    global stopTest

    # Print
    logger.info(f"Preparing for battery discharge cycle. C = {config.dischargeC} A, V = {target_v}")

    # Clear instruments
    dcl.clear()
    dmm.clear()

    # Configure the load for constant current discharge
    dcl.function = LoadFunction.ConstantCurrent
    dcl.current = config.dischargeC * 0.5

    # Configure DMM for measurement of battery voltage
    dmm.measurement = SDMMeasurement.DCV
    dmm.dcv_range = SDMVoltageRange.V_20V
    
    # Wait for instruments
    dcl.block_until_complete()
    dmm.block_until_complete()

    # Enable load
    dcl.input = True

    # Wait
    time.sleep(1)

    # Start time
    start = time.time()
    # Flag to break out of loop
    discharged: bool = False
    
    # Loop while we're not charged and haven't timed out
    while not discharged and time.time() < (start + timeout_s) and not stopTest:
        # Measure instrument readings
        dcl_c.append((time_ms(), dcl.meas_current))
        dcl_v.append((time_ms(), dcl.meas_voltage))
        dmm_v.append((time_ms(), dmm.value))
        # Calculate new averages
        dcl_c_avg = mean([item[1] for item in dcl_c[-5:]])
        dcl_v_avg = mean([item[1] for item in dcl_v[-5:]])
        dmm_v_avg = mean([item[1] for item in dmm_v[-5:]])
        # Print
        logger.info(f"  - Discharge Current: {dcl_c_avg:.3f} A, Battery Voltage: {dmm_v_avg:.3f} V")
        # Check if we hit our charge current cutoff (0.02 * C) and we have taken at least 5 samples
        if dmm_v_avg <= target_v and len(dmm_v) > 5:
            logger.info("  - Hit discharging voltage cutoff, discharged!")
            discharged = True
            break
        # Delay
        time.sleep(0.5)

    # Shut off DCL
    dcl.input = False

    if not discharged:
        logger.error("Timed out before discharging completed!")
        return False
    else:
        logger.info("Discharging complete!")
        return True

def testDCR(config: Config) -> float | None:
    """
    Test DC resistance of a battery

    Args:
        config (Config): _description_

    Returns:
        bool: True on success, False on failure
    """
    # Globals
    global psu, dcl, dmm
    global psu_v, psu_c, dcl_v, dcl_c, dmm_v
    global stopTest

    # Clear instruments
    dcl.clear()
    dmm.clear()

    # Setup DMM
    dmm.measurement = SDMMeasurement.DCV
    dmm.dcv_range = SDMVoltageRange.V_20V

    # Setup DCL for 0.25C discharge
    dcl.function = LoadFunction.ConstantCurrent
    dcl.current = 0.25 * config.dischargeC

    # Local tracking for current and voltage
    loc_dcl_c: list[float] = []
    loc_dmm_v: list[float] = []

    # Storage for overall measurements
    v_lowCurrent: float
    v_highCurrent: float

    i_lowCurrent: float
    i_highCurrent: float

    # Start low-current discharge and average 10 measurements
    dcl.input = True
    time.sleep(0.5)
    for i in range(10):
        # Check for stop test
        if stopTest:
            break
        # Take measurements
        dcl_v.append((time_ms(), dcl.meas_voltage))
        dcl_c.append((time_ms(), dcl.meas_current))
        dmm_v.append((time_ms(), dmm.value))
        # Copy to local measurements
        loc_dcl_c.append(dcl_c[-1][1])
        loc_dmm_v.append(dmm_v[-1][1])
        # Calculate averages from local storage
        dcl_c_avg = mean(loc_dcl_c[-5:])
        dmm_v_avg = mean(loc_dmm_v[-5:])
        # Print
        logger.info(f"  - DCR Baseline: {dcl_c_avg:.3f} A, {dmm_v_avg:.3f} V")
        # Sleep
        time.sleep(0.25)

    # Check stop test
    if stopTest:
        # Turn off load
        dcl.input = False
        # Return None
        return None

    # Store
    v_lowCurrent = mean(loc_dmm_v)
    i_lowCurrent = mean(loc_dcl_c)

    # Up current to 1.0C
    dcl.current = config.dischargeC

    # Clear locals
    loc_dcl_c = []
    loc_dmm_v = []

    # Take high-current measurements
    time.sleep(0.5)
    for i in range(10):
        if stopTest:
            break
        # Take measurements
        dcl_v.append((time_ms(), dcl.meas_voltage))
        dcl_c.append((time_ms(), dcl.meas_current))
        dmm_v.append((time_ms(), dmm.value))
        # Copy to local measurements
        loc_dcl_c.append(dcl_c[-1][1])
        loc_dmm_v.append(dmm_v[-1][1])
        # Calculate averages from local storage
        dcl_c_avg = mean(loc_dcl_c[-5:])
        dmm_v_avg = mean(loc_dmm_v[-5:])
        # Print
        logger.info(f"  - DCR High-Load: {dcl_c_avg:.3f} A, {dmm_v_avg:.3f} V")
        # Sleep
        time.sleep(0.25)

    # Turn off load
    dcl.input = False

    # Check stop test
    if stopTest:
        # Return None
        return None

    # Store
    v_highCurrent = mean(loc_dmm_v)
    i_highCurrent = mean(loc_dcl_c)

    # Calculate DCR
    DCR = (v_lowCurrent - v_highCurrent) / (i_highCurrent - i_lowCurrent) - config.testLeadR
    logger.info(f"Calculated DCR: ({v_lowCurrent:.3f} - {v_highCurrent:.3f} V) / ({i_highCurrent:.3f} - {i_lowCurrent:.3f} A) - {config.testLeadR} = {DCR:.3f} Ω")
    return DCR


def startTest(config: Config, window: sg.Window):
    """
    Start the battery test
    """
    global testRunning
    global stopTest
    global psu, dcl, dmm
    global psu_v, psu_c, dcl_v, dcl_c, dmm_v
    # Running
    testRunning = True
    # Blank out data
    psu_v = []
    psu_c = []
    dcl_v = []
    dcl_c = []
    dmm_v = []
    # Log
    logger.info("Preparing to run battery test")
    window.write_event_value("-TEST-STATUS-", "Initializing instruments")
    # Resource manager
    rm = ResourceManager()
    # Instantiate instruments
    psu = SPD4000X(f"TCPIP0::{config.psuIP}::inst0::INSTR", rm)
    dcl = SDL1000X(f"TCPIP0::{config.dclIP}::inst0::INSTR", rm)
    dmm = SDM3000X(f"TCPIP0::{config.dmmIP}::inst0::INSTR", rm)
    # Init
    logger.info("Resetting instruments")
    psu.reset()
    dmm.reset()
    psu.clear()
    dmm.clear()
    dcl.clear()
    psu.block_until_complete()
    dcl.block_until_complete()
    dmm.block_until_complete()

    logger.info("Instruments ready, starting battery test")
    window.write_event_value("-TEST-STATUS-", "Measuring battery voltage")

    # Measure starting battery voltage
    time.sleep(2)
    v = dmm.value
    logger.info(f"Starting battery voltage: {v:.3f} V")
    if (v < 3.20):
        logger.error(f"Battery starting voltage below safety threshold! {v:.3f} < 3.200")
        window.write_event_value("-TEST-STATUS-", "Battery undervoltage!")
        endTest(window)
        return

    # We want to be in the range of [3.6, 3.8] V for the DCR test
    low_target = config.chargeV
    high_target = config.dischargeV

    # If lower, charge
    if v < low_target:
        logger.info(f"Battery below threshold of {low_target:.3f} V, charging")
        window.write_event_value("-TEST-STATUS-", "Charging battery")
        if not charge(config, low_target, timeout_s=config.dischargeT):
            logger.error(f"Charging failed!")
            window.write_event_value("-TEST-STATUS-", "Charging failed!")
            endTest(window)
            return

    # If higher, discharge
    if v >= high_target:
        logger.info(f"Battery above threshold of {high_target:.3f} V, discharging")
        window.write_event_value("-TEST-STATUS-", "Discharging battery")
        if not discharge(config, high_target, timeout_s=config.dischargeT):
            logger.error(f"Discharging failed!")
            window.write_event_value("-TEST-STATUS-", "Discharging failed!")
            endTest(window)
            return

    # Do a pulsed current DCR test
    window.write_event_value("-TEST-STATUS-", "Testing battery DCR")
    DCR = testDCR(config)
    if not DCR:
        logger.error(f"DCR test failed!")
        window.write_event_value("-TEST-STATUS-", "DCR Measurement Failed!")
        endTest(window)
        return
    else:
        logger.info(f"DCR test complete!")
        window.write_event_value("-TEST-STATUS-", f"Test complete! DCR = {DCR:.3f} Ω")
        window.write_event_value("-DCR-VALUE-", DCR)

    # Done!
    endTest(window)
    return

def endTest(window: sg.Window):
    # Globals
    global psu, dcl, dmm
    global stopTest
    global testRunning
    
    # Ensure all sources/sinks are off
    if psu:
        for i in range(1, 5):
            psu.channel(i).output = False
    if dcl:
        dcl.input = False
    
    # No longer running
    testRunning = False

    # Send message to main window that test is done
    window.write_event_value("-TEST-ENDED-", "")

def saveConfig(config: Config):
    """
    Write the current config to an ini file

    Args:
        config (Config): the config to write
    """
    # Create config parser
    parser = ConfigParser()
    parser.optionxform = str
    # Add config section
    parser.add_section("config")
    # Iterate over dataclass
    for key,value in asdict(config).items():
        parser.set("config", key, str(value))
    # Write to file
    logger.debug(parser["config"])
    with open(configPath, "w", encoding="utf-8") as configFile:
        parser.write(configFile)
    # Log
    logger.info(f"Saved config to {configPath}")

def loadConfig() -> Config:
    """
    Read config from config.ini file, or save default config if none exists

    Returns:
        Config: the read config
    """
    logger.info(f"Loading config file {configPath}")
    if not os.path.exists(configPath):
        logger.warning("Config file does not exist, saving defaults...")
        # Write default config
        saveConfig(Config())
        # Return default config
        return Config()
    else:
        parser = ConfigParser()
        parser.optionxform = str
        parser.read(configPath)
        config = parser.parse_section(using_dataclass=Config, section_name="config")
        logger.info("Successfully read config")
        logger.debug(config)
        return config

def updateConfig(config: Config, values: Any):
    """
    Update the running config based on the current field values
    """
    config.psuIP = values["PSU_IP"]
    config.psuCh = int(values["PSU_CH"])
    config.dclIP = values["DCL_IP"]
    config.dmmIP = values["DMM_IP"]
    config.chargeC = float(values["CHG_A"])
    config.chargeV = float(values["CHG_V"])
    config.dischargeC = float(values["DSG_A"])
    config.dischargeV = float(values["DSG_V"])
    config.dischargeT = int(values["DSG_T"])
    config.dcrR = float(values["DCR_R"])
    config.testLeadR = float(values["TEST_R"])

def enableInputs(window: sg.Window, enabled: bool):
    for key in ["PSU_IP", "PSU_CH", "DCL_IP", "DMM_IP", "CHG_A", "CHG_V", "DSG_A", "DSG_V", "DSG_T", "DCR_R", "TEST_R"]:
        window[key].update(disabled = not enabled)

def showPassFail(value: float, passed: bool):
    # Get color and text
    bgColor = ""
    statusText = ""
    if passed:
        bgColor = "green"
        statusText = "BATTERY PASS"
    else:
        bgColor = "darkred"
        statusText = "BATTERY FAIL"
    # Prepare layout
    passFailLayout = [
            # Main pass/fail text
            [ sg.Text(statusText, font = ("Helvetica", 24, "bold"), key = "PASS_FAIL", expand_x=True, justification="center", background_color=bgColor) ],
            # Measurement value
            [ sg.Text(f"Battery DCR: {value:.3f} Ω", key = "MEASUREMENT", expand_x=True, justification="center", background_color=bgColor) ],
            # OK button
            [ sg.Button("OK", key="BTN_OK") ]
    ]
    # Create window
    passFailWindow = sg.Window(title = "", layout = passFailLayout, modal = True, finalize=True, background_color=bgColor, location=mainWindow.current_location())
    # Play window ding
    passFailWindow.ding()
    # Display the window
    while True:
        event, values = passFailWindow.read()
        if event == sg.WIN_CLOSED or event == "BTN_OK":
            passFailWindow.close()
            break

def close():
    """
    Window close event
    """
    sys.exit(0)

# Event loop
def main(window: sg.Window):
    # Globals
    global testRunning, stopTest
    global psu_v, psu_c, dcl_v, dcl_c, dmm_v
    # Read config
    config = loadConfig()
    # Populate fields
    window["PSU_IP"].update(config.psuIP)
    window["PSU_CH"].update(config.psuCh)
    window["DCL_IP"].update(config.dclIP)
    window["DMM_IP"].update(config.dmmIP)
    window["CHG_A"].update(config.chargeC)
    window["CHG_V"].update(config.chargeV)
    window["DSG_A"].update(config.dischargeC)
    window["DSG_V"].update(config.dischargeV)
    window["DSG_T"].update(config.dischargeT)
    window["DCR_R"].update(config.dcrR)
    window["TEST_R"].update(config.testLeadR)
    # Runtime loop
    while True:
        event, values = window.read()
        # Update config values
        updateConfig(config, values)
        # Start test on test button press
        if event == "BTN_START":
            if testRunning:
                stopTest = True
                window["BTN_START"].update(disabled = True)
            else:
                # Update button
                window["BTN_START"].update(text = "Stop")
                enableInputs(window, False)
                window.perform_long_operation(
                    lambda: startTest(config, window),
                    "-TEST-RETURNED-"
                )

        elif event == "-TEST-STATUS-":
            window["STATUS_TEXT"].update(values["-TEST-STATUS-"])
        
        elif event == "-TEST-ENDED-":
            # Update global
            stopTest = False
            # Update button
            window["BTN_START"].update(text = "Start")
            window["BTN_START"].update(disabled = False)
            enableInputs(window, True)

        elif event == "-PSU-V-":
            psu_v.append((time_ms(), float(values["-PSU-V-"])))
        elif event == "-PSU-C-":
            psu_c.append((time_ms(), float(values["-PSU-C-"])))

        elif event == "-DCL-MEAS-":
            pass

        elif event == "-DMM-MEAS-":
            pass

        elif event == "-DCR-VALUE-":
            dcr = float(values["-DCR-VALUE-"])
            # Determine pass/fail
            if dcr > config.dcrR:
                # Show failure screen
                showPassFail(dcr, False)
            else:
                # Show pass screen
                showPassFail(dcr, True)

        # Save Config Command
        elif event == "BTN_SAVECFG":
            saveConfig(config)
        
        # Exit window command
        elif event == "BTN_EXIT" or event == sg.WIN_CLOSED:
            stopTest = True
            while (testRunning):
                time.sleep(0.01)
            close()

# Startup
if __name__ == "__main__":
    main(mainWindow)