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

./build/bin/Release/

# Testing

```bash
dotnet run --project WeatherAiDotNet -- --gpu-layers 0 --ctx-size 16384 --top-k 6 --retrieval-pool 24
```

```base
dotnet run --project WeatherAiDotNet --	--gpu-layers 999 --threads 8 --batch-threads 8 --batch-size 512 --ubatch-size 256
```

```bash
dotnet run --project ./WeatherAiDotNet/WeatherAiDotNet.csproj --   --pdf-folder "./Data"   --model-path "./WeatherAiDotNet/bin/Debug/net10.0/AiModels/qwen2.5-coder-7b-instruct-q4_k_m.gguf"   --embedding-model-path "./WeatherAiDotNet/bin/Debug/net10.0/AiModels/bge-base-en-v1.5-q4_k_m.gguf"   --llama-backend vulkan   --prefer-gpu true   --gpu-layers 999   --ctx-size 8192   --threads 8   --batch-threads 8   --batch-size 512   --ubatch-size 256   --top-k 8   --retrieval-pool 48   --chunk-size 900   --chunk-overlap 180   --temperature 0.2   --top-p 0.85   --include-images true
```

Need to install Teseract then can use --ocr-cli "C:/Program Files/Tesseract-OCR/tesseract.exe"
