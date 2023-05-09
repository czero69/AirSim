# Citysample with Airsim UE5

This branch is work in progress. Update, now it works with UE 5.1 (and UE 5.2 from ue5-main branch selected commit - see below)!

Airsim limitations in UE5:
- only airsim computervision mode is supported
- record button is broken now (I am recording data with python script)

I am using this project to record data (Images, Gbuffers, ...) in Unreal Engine 5 demo for machine learning project.

:dvd: [Download example samples](https://drive.google.com/file/d/1sVwTJhNWSzcWlO9Hvi6p_RlbyFTMe6X6/view?usp=sharing)

## Install Visual Studio and tools.

My stack:
- Visual Studio 2022
- Windows 10 SDK (10.0.18362.0)

> :warning: **The version needs to be 2022**: (Airsim Repo will not accept other toolset by default when manual building). Pre-built libraries are also shipped with this toolchain and linker will not be able to match them with other versions.

## Get Unreal Engine

### UE 5.1 stock version (no Stencil support)

You need 5.1 Unreal Engine Version (worked also in 5.01-5.03 in the past), you can use marketplace ue5 build.
Note with version 5.1 and older, you will not have any segmentation masks except only humans. This is because back then nanite was not working with custom stencil buffer writing.


### Custom UE 5.2 (Stencils are supported)

- git clone ue-5 main branch from official repos https://github.com/EpicGames/UnrealEngine/tree/ue5-main 


- checkout exactly this commit. Note: it will not work with more newer commits! (Changes in Mass will not compile with current citysample) 
```bash
c825148dc6e018f358c5d36346c8698c47835a48
```

- Generate project files. For generating visual files for Visual Studio 2022 use this flag:

```bash
GenerateProjectFiles.bat -2022
```

- go to:
```bash
Engine/Plugins/Experimental/ChaosUserDataPT/Source/ChaosUserDataPT/Public/ChaosUserDataPT.h
```
and change 102 line to:
```cpp
if (const FPhysicsSolverBase* MySolver = this->GetSolver())
```

- go to:
```bash
Engine/Plugins/Runtime/MassEntity/Source/MassEntity/Public/MassRequirements.h
```
and change (comment) 199 line(s) to:
```cpp
//checkf(FragmentRequirements.FindByPredicate([](const FMassFragmentRequirementDescription& Item) { return Item.StructType == T::StaticStruct(); }) == nullptr
	//, TEXT("Duplicated requirements are not supported. %s already present"), *T::StaticStruct()->GetName());
```

- if you have compiler errors in ue5-main related to include order version, you can try for some selected something.Targeet.cs to add this line. In the end it should work without it, but for me this include ordering version is now a little buggy and random.

```c#
IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_2;
```

> :information_source: **Stencils support only starting from UE 5.2**: Writing to custom stencil depth buffers is possible starting from UE 5.2. If you want to have stencil buffers, you need UE 5.2 version with above modifications 

## A. Get modified CitySample demo with Airsim plugin compiled

Not yet available (300GB). I will share this possibly somehow in the future.

## B. Download CitySample (5.1 version) from epic store/marketplace and build Airsim plugin for your own

Now it is supported also with ue 5.2 above commit checkout. 

Build airsim plugin from this branch the same way as in official [airsim building instructions](https://microsoft.github.io/AirSim/build_windows/). After you have build a plugin from the proper command prompt with build.cmd (please go with Visual 2022 cmd), copy a plugin dir (AirSim→Unreal→Plugins) to CitySample Project plugins dir (CitySample→Plugins). 

Generate Visual studio project files (right click on .uproject, select proper UE5 version → generate). Add "airsim" to .uproject and Config\DefaultGame.ini. Follow these instructions: [add airsim to .uproject](https://microsoft.github.io/AirSim/unreal_custenv/)

When it compiles, select CitySample Project in Visual and press CTRL+F5 (run and detach). 
Then, in the editor open Map\Big_City_LVL (it will take a while).

When done, you can package a project or you can run it directly from the editor :arrow_forward:. In both cases Airsim plugin will be working. The second one is good for quick tests. Please note, that first time UE5 is running this demo, everything will take a longer time.

modify “World Settings->Selected Gamemode”. Change HUD to SimHUD.

![AirSim in UE5](readme-screenshots/gamemode_ue5_airsim.png?raw=true "select proper hud as SimHUD")

Now at this stage you should be able to run a Citysample demo from the editor or package it.

If you want to get stencils/segmentation-masks, please check (set your stencil values accordingly)
AirSim\PythonClient\computer_vision\UECapture\set_stencils_from_editor.py
and run it from unreal engine python command.

After packaging, you are ready to go wtih the next steps.

> :warning: **Record button broken - use python script**: Please note that for now red Recording airsim button is not working (will crash). This is because I didnt fix yet velococity 3 channel float buffer yet there. For now I am grabbing gbuffers and image from a python script. For now, please also use python script instead of record button.

> :warning: **Nanite stencils not yet supported in UE5.1**: For now (as of UE 5.1) nanite does not support writing to stencil buffers (see in official doc Nanite Virtualized Geometry ). (humans/character stencil works, but car/street/buidling are not writing to stencil buffers). As a workaround I am adding invisible to rgb no-nanite meshes - will update this tutorial later how I am making it, but I did it only for cars models so far.

> :information_source: **Solution for not correct VS toolchain picked up when building CitySample**: I have set confing in Citysample project now explicitly to use VS 2022. But still, if you have older Visual Studio installed in parallel, other toolchain might be choose by default. To change toolchain, open UE5, create empty project, and set toolchain for the project and for the editor in two places. 
> - Edit → Project Settings → Platforms → Windows → Toolchain → CompilerVersion → VS 2022
> - Editor Preferences →  search for “Source Code Editor” → VS 2022 (or Engine Settings → “Source Code” section. )

## Get Airsim Config File

Paste a config json file to your "Documents/Airsim" dir. This File sets up “computer vision mode”, resolutions and how buffers are saved.

My config is in my_json dir:

[settings.json](my_json/settings.json "my_json/settings.json")

## Clone modified Airsim Repo

Clone modified Airsim repo and switch to citysample branch (possibly you already have it when you followed step B “B. Download CitySample 5.1 version from epic store/marketplace”).

## How to set up python environment

Install Conda, open Anaconda Prompt

```bash
conda create --name airsim python=3.8
```

cd to Airsim root dir

```bash
conda activate airsim
cd PythonClient\computer_vision\UECapture
pip install -r requirements.txt
pip install tqdm
pip install --user numpy numpy-quaternion
python -m pip install -e ./
# when you open Modified Citysample UE5 demo with our Airsim plugin, run
python main.py --record --verbose
```

## Summary

```bash
python main.py --record
``` 

Python client should connect with compiled running ue5 citysample demo and move camera. UE5 game environment should be slowed down 100x times and you see camera teleporting and gbuffers being saved to hdd.


# Using AirSim in UE5 (OLD)

These are a bit older instructions how to use it in UE 5.01-5.1. For now, I am leaving them as is, before refactoring.

This Branch shows how I managed to run and use the AirSim plugin in UE5. The code comments unsupported vehicle setups, which allows to compile AirSim for UE5 and use it in "ComputerVision" mode. Please note that other modes: {"Multirotor", "Car"} are not supported here and will not work properly when using this branch.

What works for me so far:
- "ComputerVision" mode
- saving depth, normals, segmentation
- python api to control ue5 app

What does not work:
- opticalFlow (Hsli compilation errors - not yet investigated)
- "Multirotor" and "Car" (for now I am not bringing these as I don't need them)

### How to compile

my stack:
- Unreal Engine 5.0.2, Release branch
- Visual Studio 2022 and Windows 10 SDK (10.0.19041.0)

AirSim by default wants to be compiled with 10.0.19041.0. If you have other Windows 10 SDK versions installed, I suggest uninstalling them so Unreal Engine/Visual Studio will not pick them by accident. Get Unreal Engine 5.0.2 Release branch (or possibly newer). For the 5.0.1 Release, you will need to fix JSON 1.2 support manually.

Get code from this AirSim Branch. Compile it and add it to your Unreal Project (copy plugin dir, edit .uproject). Follow official AirSim build instructions for these.

In setting.json set:
```
"SimMode": "ComputerVision"
```

Enjoy AirSim "ComputerVision" mode in UE5!

### City Sample Demo - notes

Warning - you may need additional adjustments to your project or AirSim to make it work properly.

For example, when working with the City Sample UE5 demo I learned:
1. AirSim secondary cameras are not working with Lumen by default.
2. AirSim cameras are based on SceneComponent2D. This means they will not capture reflections outside of the camera view.
3. AirSim is setting up world origin which causes trouble for City Sample Demo. SetNewWorldOrigin() methods must be commented out in the AirSim code, as it will cause a lot of glitches (spawning low-poly buildings, not drawing small mass ai crowd/traffic, crowd/traffic wrongly placed, etc.)

As a workaround for 1. and 2. I rely on taking screenshots from the main camera as standard RGB image. Position and FOV (90deg) for the main camera and AirSim ComputerVision camera(s) are the same.

I will paste here a link to a branch when I am using mentioned fixes in City Sample Demo.

![AirSim in UE5](readme-screenshots/screen1.jpg?raw=true "AirSim depth, img, normal, cameras in Ue5. Note that AirSim camera (middle) is lacking Lumen and out of the screen reflections")



# Welcome to AirSim

AirSim is a simulator for drones, cars and more, built on [Unreal Engine](https://www.unrealengine.com/) (we now also have an experimental [Unity](https://unity3d.com/) release). It is open-source, cross platform, and supports software-in-the-loop simulation with popular flight controllers such as PX4 & ArduPilot and hardware-in-loop with PX4 for physically and visually realistic simulations. It is developed as an Unreal plugin that can simply be dropped into any Unreal environment. Similarly, we have an experimental release for a Unity plugin.

Our goal is to develop AirSim as a platform for AI research to experiment with deep learning, computer vision and reinforcement learning algorithms for autonomous vehicles. For this purpose, AirSim also exposes APIs to retrieve data and control vehicles in a platform independent way.

**Check out the quick 1.5 minute demo**

Drones in AirSim

[![AirSim Drone Demo Video](docs/images/demo_video.png)](https://youtu.be/-WfTr1-OBGQ)

Cars in AirSim

[![AirSim Car Demo Video](docs/images/car_demo_video.png)](https://youtu.be/gnz1X3UNM5Y)


## How to Get It

### Windows
[![Build Status](https://github.com/microsoft/AirSim/actions/workflows/test_windows.yml/badge.svg)](https://github.com/microsoft/AirSim/actions/workflows/test_windows.yml)
* [Download binaries](https://github.com/Microsoft/AirSim/releases)
* [Build it](https://microsoft.github.io/AirSim/build_windows)

### Linux
[![Build Status](https://github.com/microsoft/AirSim/actions/workflows/test_ubuntu.yml/badge.svg)](https://github.com/microsoft/AirSim/actions/workflows/test_ubuntu.yml)
* [Download binaries](https://github.com/Microsoft/AirSim/releases)
* [Build it](https://microsoft.github.io/AirSim/build_linux)

### macOS
[![Build Status](https://github.com/microsoft/AirSim/actions/workflows/test_macos.yml/badge.svg)](https://github.com/microsoft/AirSim/actions/workflows/test_macos.yml)
* [Build it](https://microsoft.github.io/AirSim/build_macos)

For more details, see the [use precompiled binaries](docs/use_precompiled.md) document. 

## How to Use It

### Documentation

View our [detailed documentation](https://microsoft.github.io/AirSim/) on all aspects of AirSim.

### Manual drive

If you have remote control (RC) as shown below, you can manually control the drone in the simulator. For cars, you can use arrow keys to drive manually.

[More details](https://microsoft.github.io/AirSim/remote_control)

![record screenshot](docs/images/AirSimDroneManual.gif)

![record screenshot](docs/images/AirSimCarManual.gif)


### Programmatic control

AirSim exposes APIs so you can interact with the vehicle in the simulation programmatically. You can use these APIs to retrieve images, get state, control the vehicle and so on. The APIs are exposed through the RPC, and are accessible via a variety of languages, including C++, Python, C# and Java.

These APIs are also available as part of a separate, independent cross-platform library, so you can deploy them on a companion computer on your vehicle. This way you can write and test your code in the simulator, and later execute it on the real vehicles. Transfer learning and related research is one of our focus areas.

Note that you can use [SimMode setting](https://microsoft.github.io/AirSim/settings#simmode) to specify the default vehicle or the new [ComputerVision mode](https://microsoft.github.io/AirSim/image_apis#computer-vision-mode-1) so you don't get prompted each time you start AirSim.

[More details](https://microsoft.github.io/AirSim/apis)

### Gathering training data

There are two ways you can generate training data from AirSim for deep learning. The easiest way is to simply press the record button in the lower right corner. This will start writing pose and images for each frame. The data logging code is pretty simple and you can modify it to your heart's content.

![record screenshot](docs/images/record_data.png)

A better way to generate training data exactly the way you want is by accessing the APIs. This allows you to be in full control of how, what, where and when you want to log data.

### Computer Vision mode

Yet another way to use AirSim is the so-called "Computer Vision" mode. In this mode, you don't have vehicles or physics. You can use the keyboard to move around the scene, or use APIs to position available cameras in any arbitrary pose, and collect images such as depth, disparity, surface normals or object segmentation.

[More details](https://microsoft.github.io/AirSim/image_apis)

### Weather Effects

Press F10 to see various options available for weather effects. You can also control the weather using [APIs](https://microsoft.github.io/AirSim/apis#weather-apis). Press F1 to see other options available.

![record screenshot](docs/images/weather_menu.png)

## Tutorials

- [Video - Setting up AirSim with Pixhawk Tutorial](https://youtu.be/1oY8Qu5maQQ) by Chris Lovett
- [Video - Using AirSim with Pixhawk Tutorial](https://youtu.be/HNWdYrtw3f0) by Chris Lovett
- [Video - Using off-the-self environments with AirSim](https://www.youtube.com/watch?v=y09VbdQWvQY) by Jim Piavis
- [Webinar - Harnessing high-fidelity simulation for autonomous systems](https://note.microsoft.com/MSR-Webinar-AirSim-Registration-On-Demand.html) by Sai Vemprala
- [Reinforcement Learning with AirSim](https://microsoft.github.io/AirSim/reinforcement_learning) by Ashish Kapoor
- [The Autonomous Driving Cookbook](https://aka.ms/AutonomousDrivingCookbook) by Microsoft Deep Learning and Robotics Garage Chapter
- [Using TensorFlow for simple collision avoidance](https://github.com/simondlevy/AirSimTensorFlow) by Simon Levy and WLU team

## Participate

### Paper

More technical details are available in [AirSim paper (FSR 2017 Conference)](https://arxiv.org/abs/1705.05065). Please cite this as:
```
@inproceedings{airsim2017fsr,
  author = {Shital Shah and Debadeepta Dey and Chris Lovett and Ashish Kapoor},
  title = {AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles},
  year = {2017},
  booktitle = {Field and Service Robotics},
  eprint = {arXiv:1705.05065},
  url = {https://arxiv.org/abs/1705.05065}
}
```

### Contribute

Please take a look at [open issues](https://github.com/microsoft/airsim/issues) if you are looking for areas to contribute to.

* [More on AirSim design](https://microsoft.github.io/AirSim/design)
* [More on code structure](https://microsoft.github.io/AirSim/code_structure)
* [Contribution Guidelines](CONTRIBUTING.md)

### Who is Using AirSim?

We are maintaining a [list](https://microsoft.github.io/AirSim/who_is_using) of a few projects, people and groups that we are aware of. If you would like to be featured in this list please [make a request here](https://github.com/microsoft/airsim/issues).

## Contact

Join our [GitHub Discussions group](https://github.com/microsoft/AirSim/discussions) to stay up to date or ask any questions.

We also have an AirSim group on [Facebook](https://www.facebook.com/groups/1225832467530667/). 


## What's New

* [Cinematographic Camera](https://github.com/microsoft/AirSim/pull/3949)
* [ROS2 wrapper](https://github.com/microsoft/AirSim/pull/3976)
* [API to list all assets](https://github.com/microsoft/AirSim/pull/3940)
* [movetoGPS API](https://github.com/microsoft/AirSim/pull/3746)
* [Optical flow camera](https://github.com/microsoft/AirSim/pull/3938)
* [simSetKinematics API](https://github.com/microsoft/AirSim/pull/4066)
* [Dynamically set object textures from existing UE material or texture PNG](https://github.com/microsoft/AirSim/pull/3992)
* [Ability to spawn/destroy lights and control light parameters](https://github.com/microsoft/AirSim/pull/3991)
* [Support for multiple drones in Unity](https://github.com/microsoft/AirSim/pull/3128)
* [Control manual camera speed through the keyboard](https://github.com/microsoft/AirSim/pulls?page=6&q=is%3Apr+is%3Aclosed+sort%3Aupdated-desc#:~:text=1-,Control%20manual%20camera%20speed%20through%20the%20keyboard,-%233221%20by%20saihv) 

For complete list of changes, view our [Changelog](docs/CHANGELOG.md)

## FAQ

If you run into problems, check the [FAQ](https://microsoft.github.io/AirSim/faq) and feel free to post issues in the  [AirSim](https://github.com/Microsoft/AirSim/issues) repository.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## License

This project is released under the MIT License. Please review the [License file](LICENSE) for more details.


