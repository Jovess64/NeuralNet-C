## Project description
A simple fully-connected neural network implemented from scratch in C, supporting custom network architecture configuration. Trains on image datasets (such as MNIST, which is provided by default) and demonstrates basic forward and backward propagation without using any external libraries.

## Motivation
This project is a learning exercise to understand the internals of neural networks by implementing them from scratch in C, gaining insight into frameworks like TensorFlow.
It was also an excellent opportunity to practise writing portable C code.

## Features
- **Optimisation method:** Stochastic gradient descent
- **Cost function:** Total squared error
- **Activation functions:**
	- **Hidden layers:** ReLU
	- **Output layer:** Logistic sigmoid function
- **Learning rate scheduler:** Exponential decay
- **Hardware support:** CPU-only, single-threaded

## Build instructions
**NOTE: this project requires C99 or newer**

#### Manual compilation:
```bash
gcc src/main.c src/fileHandling.c src/helpers.c -lm -o cnn -O3 -march=native -ffast-math -flto -s
```
- The resulting executable will be in the root directory
- To use a different compiler, just replace the `gcc` command with the correct command for your chosen compiler

#### CMake:
```bash
mkdir build
cd build
cmake -G "[GENERATOR]" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cmake --install . --prefix ../final_build
```
- Replace `[GENERATOR]` with the name of your code generator, such as `MinGW Makefiles`, `Unix Makefiles`, `Ninja`, and `Visual Studio 17 2022`
- To use a different compiler, use `-DCMAKE_C_COMPILER=[COMPILER]` during CMake configuration
	- Replace `[COMPILER]` with the path of your C compiler of choice, such as `gcc`
- **For multi-configuration generators such as Visual Studio or Xcode:** remove `-DCMAKE_BUILD_TYPE=Release`, and add `--config Release` during both building and resource installation
	- **For MSVC:** manually specify the version with `-T` during CMake configuration, for example, `-T v143`. Use any version so long as it supports all utilised features (NOTE: MSVC features do not always conform to the C standard)
- **For GCC/Clang:** `-DUSE_NATIVE_CPU=OFF` can be added during CMake configuration to disable CPU-specific optimisations, which are enabled by default
- The resulting built executable and installed resources will all be in `./final_build/`
	- Replacing `../final_build` during installation will put resources there instead

## Config file
Specifies network and program parameters before runtime.

Path is defined as `CONFIG_FILENAME` in `src/main.h` (`"config.cfg"` by default), and format is codified manually in `src/config_context.h` (specifically `struct GetConfigContext`) and `fileHandling.c` (specifically `int GetConfig()`).

A default config is provided in `config.cfg`.

### Format
- Each item on its own line
- Make item empty to allow determination during runtime
- Numeric values must be in decimal format

| Parameter                | Description                                |
| ------------------------ | ------------------------------------------ |
| Learning rate            | Initial learning rate for SGD              |
| Learning rate multiplier | For exponential decay of learning rate     |
| Layers count             | Number of layers in the network            |
| Layer sizes              | Number of neurons per layer (one per line) |
| Training images path     | Path to training images                    |
| Training labels path     | Path to training labels                    |
| Testing images path      | Path to testing images                     |
| Testing labels path      | Path to testing labels                     |

## Licence
This project is open-source and available under the [MIT License](LICENSE).