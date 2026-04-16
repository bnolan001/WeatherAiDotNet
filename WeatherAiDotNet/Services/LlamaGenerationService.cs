using System.Diagnostics;
using System.Text;

namespace WeatherAiDotNet.Services;

internal static class LlamaGenerationService
{
    public static async Task<string> GenerateAnswerAsync(string llamaCliPath, string modelPath, string prompt, int maxTokens, int gpuLayers, int contextSize)
    {
        var promptFilePath = Path.Combine(Path.GetTempPath(), $"rag-prompt-{Guid.NewGuid():N}.txt");
        await File.WriteAllTextAsync(promptFilePath, prompt, Encoding.UTF8);

        var processStartInfo = new ProcessStartInfo
        {
            FileName = llamaCliPath,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        processStartInfo.ArgumentList.Add("-m");
        processStartInfo.ArgumentList.Add(modelPath);
        processStartInfo.ArgumentList.Add("-ngl");
        processStartInfo.ArgumentList.Add(Math.Max(0, gpuLayers).ToString());
        processStartInfo.ArgumentList.Add("-c");
        processStartInfo.ArgumentList.Add(Math.Max(512, contextSize).ToString());
        processStartInfo.ArgumentList.Add("-n");
        processStartInfo.ArgumentList.Add(maxTokens.ToString());
        processStartInfo.ArgumentList.Add("--no-display-prompt");
        processStartInfo.ArgumentList.Add("-f");
        processStartInfo.ArgumentList.Add(promptFilePath);
        processStartInfo.ArgumentList.Add("--simple-io");
        processStartInfo.ArgumentList.Add("--log-disable");

        using var process = new Process { StartInfo = processStartInfo };

        try
        {
            process.Start();
        }
        catch (Exception ex)
        {
            TryDelete(promptFilePath);
            return $"Unable to start llama.cpp CLI. Configure --llama-cli correctly. Error: {ex.Message}";
        }

        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();

        await process.WaitForExitAsync();

        var output = await outputTask;
        var error = await errorTask;

        TryDelete(promptFilePath);

        if (process.ExitCode != 0)
        {
            return $"llama.cpp CLI failed (exit code {process.ExitCode}).\nSTDERR:\n{error}\nSTDOUT:\n{output}";
        }

        return string.IsNullOrWhiteSpace(output)
            ? "No response returned by local model."
            : output.Trim();
    }

    private static void TryDelete(string filePath)
    {
        try
        {
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }
        catch
        {
        }
    }
}