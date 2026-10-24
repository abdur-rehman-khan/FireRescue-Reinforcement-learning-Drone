"""The deterministic default scenario: a 20 m x 14 m single-storey building.

         x=0                                              x=20
    z=14 +-----------+--------------------------------------+
         |  D        |  E   (fire F3, victim V5)            |
         |  (V4)     |                                      |
    z=8.5+--[ ]------+----------[ ]-------------------------+
         |            corridor (drone start)                 |
    z=5.5+--[ ]-------------[ ]------------------[ ]---------+
         |  A (V1)   [ ]  B (fire F1, V2)  [ ]  C (F2, V3)   |
    z=0  +-----------+-----------------+----------------------+
                     x=7              x=14
"""
from __future__ import annotations

from .config import (BuildingConfig, DroneConfig, FireConfig, FurnitureConfig, MissionConfig,
                     RoomConfig, ScenarioConfig, SensorConfig, VictimConfig, WallConfig,
                     WindowConfig)


def default_scenario() -> ScenarioConfig:
    rooms = [
        RoomConfig("Room A", (0, 0), (7, 5.5), "room", (0.78, 0.72, 0.60)),
        RoomConfig("Room B", (7, 0), (14, 5.5), "room", (0.62, 0.66, 0.72)),
        RoomConfig("Room C", (14, 0), (20, 5.5), "room", (0.70, 0.76, 0.64)),
        RoomConfig("Corridor", (0, 5.5), (20, 8.5), "corridor", (0.58, 0.58, 0.60)),
        RoomConfig("Room D", (0, 8.5), (9, 14), "room", (0.74, 0.66, 0.70)),
        RoomConfig("Room E", (9, 8.5), (20, 14), "room", (0.76, 0.72, 0.62)),
    ]
    walls = [
        # exterior shell
        WallConfig("x", 0.0, -0.1, 20.1, exterior=True),
        WallConfig("x", 14.0, -0.1, 20.1, exterior=True),
        WallConfig("z", 0.0, 0.0, 14.0, exterior=True),
        WallConfig("z", 20.0, 0.0, 14.0, exterior=True),
        # corridor walls with doorways (leaves swing away from the corridor, except north wall)
        WallConfig("x", 5.5, 0.0, 20.0, gaps=[(3.5, 1.4), (10.5, 1.4), (17.0, 1.4)], leaf_dir=-1),
        WallConfig("x", 8.5, 0.0, 20.0, gaps=[(4.5, 1.4), (14.5, 1.4)], leaf_dir=1),
        # dividing walls between rooms
        WallConfig("z", 7.0, 0.0, 5.5, gaps=[(2.5, 1.4)], leaf_dir=-1),
        WallConfig("z", 14.0, 0.0, 5.5, gaps=[(2.5, 1.4)], leaf_dir=1),
        WallConfig("z", 9.0, 8.5, 14.0, gaps=[(11.0, 1.4)], leaf_dir=-1),
    ]
    windows = [
        WindowConfig("south", 3.5), WindowConfig("south", 10.5), WindowConfig("south", 17.0),
        WindowConfig("north", 4.5), WindowConfig("north", 14.5),
        WindowConfig("west", 2.8), WindowConfig("west", 11.2),
        WindowConfig("east", 2.8), WindowConfig("east", 11.2),
    ]
    F = FurnitureConfig
    furniture = [
        # Room A
        F("table", (4.6, 0.0, 1.5), name="tableA"),
        F("chair", (3.7, 0.0, 1.5), 90, name="chairA1"),
        F("chair", (4.6, 0.0, 2.3), 180, name="chairA2"),
        F("box", (0.7, 0.0, 4.7), name="boxA1"),
        F("box", (0.7, 0.0, 4.0), name="boxA2"),
        F("beam", (5.9, 0.0, 3.9), 90, name="beamA"),
        # Room B
        F("table", (12.4, 0.0, 4.2), name="tableB"),
        F("rubble", (8.2, 0.0, 4.2), name="rubbleB"),
        F("beam", (8.1, 0.0, 0.9), name="beamB"),
        F("box", (13.3, 0.0, 0.8), name="boxB"),
        # Room C
        F("table", (16.0, 0.0, 3.6), 90, name="tableC"),
        F("chair", (16.9, 0.0, 3.6), 270, name="chairC"),
        F("box", (19.3, 0.0, 0.7), name="boxC1"),
        F("box", (19.3, 0.0, 4.9), name="boxC2"),
        # Corridor (note the hanging beam, low enough to hit if you fly high)
        F("box", (6.5, 0.0, 8.0), name="boxCorr"),
        F("beam", (11.0, 0.0, 6.0), name="beamCorr"),
        F("beam", (15.0, 1.7, 7.4), 0, (0.3, 0.3, 1.6), name="hangingBeam"),
        # Room D
        F("table", (6.5, 0.0, 12.5), name="tableD"),
        F("chair", (5.6, 0.0, 12.5), 90, name="chairD"),
        F("box", (0.7, 0.0, 13.2), name="boxD"),
        F("rubble", (7.8, 0.0, 9.6), name="rubbleD"),
        # Room E
        F("table", (11.8, 0.0, 12.6), name="tableE"),
        F("chair", (11.8, 0.0, 11.7), 0, name="chairE"),
        F("beam", (11.0, 0.0, 9.7), name="beamE"),
        F("box", (19.3, 0.0, 13.2), name="boxE1"),
        F("box", (19.3, 0.0, 12.4), name="boxE2"),
    ]
    building = BuildingConfig(rooms=rooms, walls=walls, windows=windows, furniture=furniture)
    return ScenarioConfig(
        name="default",
        building=building,
        drone=DroneConfig(start_position=(1.5, 1.2, 7.0), start_yaw_deg=90.0),
        sensor=SensorConfig(),
        mission=MissionConfig(),
        fires=[
            FireConfig("F1", (10.5, 0.4, 2.3), 2.0),
            FireConfig("F2", (17.5, 0.4, 1.9), 1.6),
            FireConfig("F3", (15.5, 0.4, 12.0), 2.0),
        ],
        victims=[
            VictimConfig("V1", (2.2, 0.0, 2.8), "standing"),
            VictimConfig("V2", (12.6, 0.0, 2.6), "prone"),
            VictimConfig("V3", (18.4, 0.0, 3.4), "prone"),
            VictimConfig("V4", (2.5, 0.0, 12.0), "prone"),
            VictimConfig("V5", (18.3, 0.0, 10.2), "standing"),
        ],
    )
