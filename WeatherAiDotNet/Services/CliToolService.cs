using System.Diagnostics;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

internal static class CliToolService
{
    public static async Task<ToolCheckResult> ValidateAsync(string executablePath, params string[] args)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = executablePath,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        foreach (var arg in args)
        {
            startInfo.ArgumentList.Add(arg);
        }

        using var process = new Process { StartInfo = startInfo };

        try
        {
            process.Start();
        }
        catch (Exception ex)
        {
            return new ToolCheckResult(false, ex.Message);
        }

        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));

        try
        {
            var outputTask = process.StandardOutput.ReadToEndAsync();
            var errorTask = process.StandardError.ReadToEndAsync();

            await process.WaitForExitAsync(timeout.Token);

            var output = await outputTask;
            var error = await errorTask;

            if (process.ExitCode == 0)
            {
                return new ToolCheckResult(true, string.Empty);
            }

            var message = string.IsNullOrWhiteSpace(error) ? output : error;
            return new ToolCheckResult(false, string.IsNullOrWhiteSpace(message)
                ? $"Process exited with code {process.ExitCode}."
                : message.Trim());
        }
        catch (OperationCanceledException)
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
            }

            return new ToolCheckResult(false, "Tool validation timed out.");
        }
    }
}