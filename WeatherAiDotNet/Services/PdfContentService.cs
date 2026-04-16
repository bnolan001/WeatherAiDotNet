using System.Diagnostics;
using System.Text;
using UglyToad.PdfPig;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

internal static class PdfContentService
{
    public static string ReadPdfText(string pdfPath)
    {
        var sb = new StringBuilder();
        using var document = PdfDocument.Open(pdfPath);

        foreach (var page in document.GetPages())
        {
            sb.AppendLine(page.Text);
        }

        return sb.ToString();
    }

    public static List<PdfImageItem> ExtractPdfImageItems(string pdfPath, string pdfRoot, string imagesOutputPath, string ocrCliPath)
    {
        var items = new List<PdfImageItem>();
        var relativePdfPath = Path.GetRelativePath(pdfRoot, pdfPath);
        var pdfRelativeWithoutExtension = Path.Combine(
            Path.GetDirectoryName(relativePdfPath) ?? string.Empty,
            Path.GetFileNameWithoutExtension(relativePdfPath));

        using var document = PdfDocument.Open(pdfPath);

        foreach (var page in document.GetPages())
        {
            var imageNumber = 0;
            foreach (var image in page.GetImages())
            {
                imageNumber++;
                if (!image.TryGetPng(out var pngBytes))
                {
                    continue;
                }

                var imageRelativePath = Path.Combine(pdfRelativeWithoutExtension, $"page-{page.Number:D4}-image-{imageNumber:D3}.png");
                var imageFullPath = Path.Combine(imagesOutputPath, imageRelativePath);

                var imageDirectory = Path.GetDirectoryName(imageFullPath);
                if (!string.IsNullOrWhiteSpace(imageDirectory))
                {
                    Directory.CreateDirectory(imageDirectory);
                }

                File.WriteAllBytes(imageFullPath, pngBytes);

                var ocrText = TryRunOcr(ocrCliPath, imageFullPath);
                var relativeImagePath = Path.GetRelativePath(Environment.CurrentDirectory, imageFullPath);

                var indexText = string.IsNullOrWhiteSpace(ocrText)
                    ? $"[PDF_IMAGE] file={relativePdfPath}; page={page.Number}; image={imageNumber}; path={relativeImagePath}; no_ocr_text_available"
                    : $"[PDF_IMAGE] file={relativePdfPath}; page={page.Number}; image={imageNumber}; path={relativeImagePath}; ocr_text={ocrText}";

                items.Add(new PdfImageItem(relativeImagePath, page.Number, imageNumber, indexText));
            }
        }

        return items;
    }

    public static IEnumerable<string> ChunkText(string text, int chunkSize, int overlap)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            yield break;
        }

        var normalized = text.Replace("\r", " ").Replace("\n", " ");
        var words = normalized.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        if (words.Length == 0)
        {
            yield break;
        }

        var chunkWordCount = Math.Max(40, chunkSize / 6);
        var overlapWordCount = Math.Max(5, overlap / 6);
        var step = Math.Max(1, chunkWordCount - overlapWordCount);

        for (var i = 0; i < words.Length; i += step)
        {
            var length = Math.Min(chunkWordCount, words.Length - i);
            var chunk = string.Join(' ', words, i, length).Trim();

            if (!string.IsNullOrWhiteSpace(chunk))
            {
                yield return chunk;
            }

            if (i + length >= words.Length)
            {
                yield break;
            }
        }
    }

    private static string TryRunOcr(string ocrCliPath, string imagePath)
    {
        if (string.IsNullOrWhiteSpace(ocrCliPath))
        {
            return string.Empty;
        }

        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = ocrCliPath,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            startInfo.ArgumentList.Add(imagePath);
            startInfo.ArgumentList.Add("stdout");
            startInfo.ArgumentList.Add("--psm");
            startInfo.ArgumentList.Add("6");

            using var process = new Process { StartInfo = startInfo };
            process.Start();

            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();

            return process.ExitCode == 0 ? NormalizeWhitespace(output) : string.Empty;
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string NormalizeWhitespace(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return string.Empty;
        }

        var builder = new StringBuilder(input.Length);
        var wasWhiteSpace = false;

        foreach (var ch in input)
        {
            if (char.IsWhiteSpace(ch))
            {
                if (!wasWhiteSpace)
                {
                    builder.Append(' ');
                    wasWhiteSpace = true;
                }
            }
            else
            {
                builder.Append(ch);
                wasWhiteSpace = false;
            }
        }

        return builder.ToString().Trim();
    }
}