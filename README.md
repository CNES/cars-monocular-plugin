# Monocular abilities for CARS

This plugin enables the use of the MoGe2 model in CARS, for higher accuracy building reconstruction.

- [Context](#contexte)
- [Installation](#installation)
- [Using the new pipeline](#using-the-new-pipeline) 

## Context

This plugin is made to be used in conjunction with CARS, the stereo-reconstruction software. 
More information can be found over at [CARS's GitHub page](https://github.com/CNES/cars).

## Installation

First clone this repository, using :

```bash
$ git clone --recurse-submodules git@github.com:CNES/cars-monocular-plugin.git
$ cd cars-monocular-plugin   
``` 

> **Note:** `--recurse-submodules` is required. This project vendors [MoGe](https://github.com/microsoft/MoGe) as a submodule, and MoGe itself vendors `utils3d` and `pipeline` as nested submodules. A plain `git clone` will leave those directories empty, causing build failures.

If you already cloned the repository without submodules, initialize them before installing:

```bash
$ git submodule update --init --recursive
```

You can then create a virtual environment and install the plugin, which will install CARS automatically :

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ make install   
``` 

Or install the plugin in your own environment, if it already has CARS :

```bash
$ source your/own/env/activate
$ pip install .
```

Once installed, don't forget to download a MoGe2 model, for example Ruicheng/moge-2-vitl-normal, using this command :

```bash
$ cars-download-moge2 --model vitl-normal
```

Or via any other means if you don't have a direct access to the internet. The plugin will attempt to download the vitl-normal model on first import, providing at least the default model.
If working from an environment such as the TREX cluster, an option is to directly use wget to fetch the model file, 
then move it to its proper place for the plugin to recognize it :

```bash
# fetch the model
$ wget https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt

# move the model to the right place
# it should be under cars_monocular_plugin/applications/depth_map_generation/models with the proper name for each model :
#  - moge-2-vitl-normal.pt
#  - moge-2-vitb-normal.pt
#  - moge-2-vits-normal.pt
$ mkdir [your/plugin/installation/path/]cars_monocular_plugin/applications/depth_map_generation/models
$ mv ./model.pt [your/plugin/installation/path/]cars_monocular_plugin/applications/depth_map_generation/models/moge-2-vitl-normal.pt
```

## Using the new pipeline

Though this pipeline is intended to be used within CARS's meta pipeline, it can still be used as a stand-alone pipeline by providing the right configuration.

Once your configuration file is ready, you can launch the pipeline using CARS : 

```bash
$ cars configfile.yaml
```

### Configuration

The monocular pipeline can be enabled by setting the pipeline parameter in the global advanced section of the CARS configuration.

A minimal example configuration is shown below:

```yaml
input:
  sensors:
    one: # sensor image path
    two: # sensor image path
pipeline: monocular
output:
  directory: outresults
```

The pipeline operates on image pairs. By default, monocular is only computed where required by downstream applications, meaning on the left images only.

Additional options specific to the monocular pipeline can be configured under the monocular section. 
For example, monocular can also be applied to right images, and the MoGe2-based depth map generation application can be configured as follows:

```yaml
input: ...
advanced: ...
output: ...
monocular:
  advanced:
    save_intermediate_data: false
    right_image_monocular: true
  applications:
    depth_map_generation:
      method: moge2
      model: Ruicheng/moge-2-vitl-normal
      save_intermediate_data: true
      edge_threshold: 0.7
```

The ``model`` parameter can reference either a local MoGe2 checkpoint or a Hugging Face model identifier.

If ``save_intermediate_data`` is set to false, only the edge map will be created in the output folder. 
Else, all by-products (depth map, normal map, tile_id) will be saved in the ``dump_dir`` folder.
