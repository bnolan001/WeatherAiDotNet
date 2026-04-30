using System.Diagnostics;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

/// <summary>
/// Validates that an external command-line tool can be launched and returns a
/// successful exit code before the application proceeds.
/// </summary>
/// <remarks>
/// This is used at startup to check optional tools such as the OCR binary so that
/// problems are surfaced early with a clear message rather than silently failing
/// during the indexing loop.
/// </remarks>
public static class CliToolService
{
    /// <summary>
    /// Launches <paramref name="executablePath"/> with the given arguments and waits
    /// up to 10 seconds for it to exit.  A zero exit code is treated as success.
    /// </summary>
    /// <param name="executablePath">Full path to the executable to validate.</param>
    /// <param name="args">
    /// Arguments to pass, e.g. <c>"--version"</c>.  Using <c>ArgumentList</c>
    /// rather than <c>Arguments</c> avoids shell-escaping issues on Windows.
    /// </param>
    /// <returns>
    /// A <see cref="ToolCheckResult"/> whose <c>Success</c> flag is <see langword="true"/>
    /// when the process exits with code 0 within the timeout, or <see langword="false"/>
    /// with an explanatory <c>Message</c> otherwise.
    /// </returns>
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
            // The executable was not found or the OS denied the launch.
            return new ToolCheckResult(false, ex.Message);
        }

        // Cancel the wait after 10 seconds so a hanging tool doesn't stall startup.
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));

        try
        {
            // Read stdout and stderr concurrently while waiting for exit to avoid
            // deadlocking when the process fills its output buffer.
            var outputTask = process.StandardOutput.ReadToEndAsync();
            var errorTask = process.StandardError.ReadToEndAsync();

            await process.WaitForExitAsync(timeout.Token);

            var output = await outputTask;
            var error = await errorTask;

            if (process.ExitCode == 0)
            {
                return new ToolCheckResult(true, string.Empty);
            }

            // Prefer stderr for the error message; fall back to stdout if stderr is empty.
            var message = string.IsNullOrWhiteSpace(error) ? output : error;
            return new ToolCheckResult(false, string.IsNullOrWhiteSpace(message)
                ? $"Process exited with code {process.ExitCode}."
                : message.Trim());
        }
        catch (OperationCanceledException)
        {
            // Kill the timed-out process so it doesn't linger in the background.
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // Ignore kill failures (race condition where process just exited).
            }

            return new ToolCheckResult(false, "Tool validation timed out.");
        }
    }
}