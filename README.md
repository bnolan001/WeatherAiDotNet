# WeatherAiDotNet

# Setup
```bash
winget install --id UB-Mannheim.TesseractOCR -e
winget install llama.cpp
```

## llama-embedding.exe generation
1: Run the below command to clone the official repository or alternatively get it from the download page.

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

2: Now build llama.cpp:

mkdir build
cd build

cmake ..
cmake --build . --config Release

Once the build has completed, you will find your executable files usually in:

.\build\bin\Release\
# Testing
```bash
dotnet run --project WeatherAiDotNet -- --gpu-layers 0 --ctx-size 16384 --top-k 6 --retrieval-pool 24
```
