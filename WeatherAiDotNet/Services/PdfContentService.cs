using System.Diagnostics;
using System.Text;
using UglyToad.PdfPig;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

/// <summary>
/// Handles all PDF content extraction tasks: reading page text, splitting it into
/// overlapping chunks suitable for embedding, extracting embedded images, and
/// optionally running OCR on those images so their text can also be indexed.
/// </summary>
public static class PdfContentService
{
    /// <summary>
    /// Opens a PDF file and concatenates the plain text from every page into a
    /// single string.  The text is extracted by PdfPig and retains approximate
    /// page ordering, but formatting (columns, tables) is not preserved.
    /// </summary>
    /// <param name="pdfPath">Absolute path to the PDF file to read.</param>
    /// <returns>All page text joined by newlines.</returns>
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

    /// <summary>
    /// Extracts every embedded raster image from a PDF, writes each one to disk as
    /// a PNG file, and optionally runs OCR on it to produce indexable text.
    /// </summary>
    /// <param name="pdfPath">Absolute path to the source PDF.</param>
    /// <param name="pdfRoot">
    /// Root folder for PDFs; used to compute a relative path for metadata tags.
    /// </param>
    /// <param name="imagesOutputPath">
    /// Folder where extracted PNG files are saved.  Sub-directories mirror the
    /// relative PDF structure so images from different PDFs stay separate.
    /// </param>
    /// <param name="ocrCliPath">
    /// Path to a Tesseract-compatible CLI (e.g., <c>tesseract.exe</c>), or an empty
    /// string to skip OCR.
    /// </param>
    /// <returns>
    /// A list of <see cref="PdfImageItem"/> records, one per extracted image.  Each
    /// record contains metadata and any OCR text, ready for embedding and indexing.
    /// </returns>
    public static List<PdfImageItem> ExtractPdfImageItems(string pdfPath, string pdfRoot, string imagesOutputPath, string ocrCliPath)
    {
        var items = new List<PdfImageItem>();
        var relativePdfPath = Path.GetRelativePath(pdfRoot, pdfPath);

        // Build a sub-folder path that mirrors the PDF's location, e.g.
        // "SubFolder/MyDocument" so images from each PDF get their own directory.
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

                // Skip images that cannot be decoded to PNG bytes (e.g., JBIG2, CMYK).
                if (!image.TryGetPng(out var pngBytes))
                {
                    continue;
                }

                // Build a deterministic file name using page and image numbers so
                // re-indexing always overwrites the same file rather than accumulating duplicates.
                var imageRelativePath = Path.Combine(pdfRelativeWithoutExtension, $"page-{page.Number:D4}-image-{imageNumber:D3}.png");
                var imageFullPath = Path.Combine(imagesOutputPath, imageRelativePath);

                var imageDirectory = Path.GetDirectoryName(imageFullPath);
                if (!string.IsNullOrWhiteSpace(imageDirectory))
                {
                    Directory.CreateDirectory(imageDirectory);
                }

                File.WriteAllBytes(imageFullPath, pngBytes);

                // Attempt OCR; returns empty string if ocrCliPath is blank or OCR fails.
                var ocrText = TryRunOcr(ocrCliPath, imageFullPath);
                var relativeImagePath = Path.GetRelativePath(Environment.CurrentDirectory, imageFullPath);

                // Compose the text that will be stored in the vector database.
                // Structured metadata tags make it easier for the model to cite the
                // source of image-derived context in its answers.
                var indexText = string.IsNullOrWhiteSpace(ocrText)
                    ? $"[PDF_IMAGE] file={relativePdfPath}; page={page.Number}; image={imageNumber}; path={relativeImagePath}; no_ocr_text_available"
                    : $"[PDF_IMAGE] file={relativePdfPath}; page={page.Number}; image={imageNumber}; path={relativeImagePath}; ocr_text={ocrText}";

                items.Add(new PdfImageItem(relativeImagePath, page.Number, imageNumber, indexText));
            }
        }

        return items;
    }

    /// <summary>
    /// Splits a long text string into overlapping word-based chunks suitable for
    /// embedding and storage in the vector database.
    /// </summary>
    /// <remarks>
    /// Overlap between consecutive chunks prevents important sentences from being
    /// split across two chunks with no shared context, which would hurt retrieval
    /// accuracy.  The word-count approach is preferred over character counts because
    /// embedding models have token limits, and words are a better proxy for tokens
    /// than raw characters.
    /// </remarks>
    /// <param name="text">The full document text to split.</param>
    /// <param name="chunkSize">
    /// Approximate target character length per chunk.  Internally converted to a
    /// word count using a heuristic of ~6 characters per word.
    /// </param>
    /// <param name="overlap">
    /// Approximate character length of the overlap region between adjacent chunks.
    /// Converted to a word count the same way as <paramref name="chunkSize"/>.
    /// </param>
    /// <returns>A lazy sequence of non-empty chunk strings.</returns>
    public static IEnumerable<string> ChunkText(string text, int chunkSize, int overlap)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            yield break;
        }

        // Collapse all whitespace so word splitting is consistent regardless of
        // whether the PDF had tabs, multiple spaces, or Windows-style line endings.
        var normalized = text.Replace("\r", " ").Replace("\n", " ");
        var words = normalized.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        if (words.Length == 0)
        {
            yield break;
        }

        // Convert character-based sizes to word counts (roughly 6 chars/word).
        // Floor values prevent degenerate zero-word chunks on very small inputs.
        var chunkWordCount = Math.Max(40, chunkSize / 6);
        var overlapWordCount = Math.Max(5, overlap / 6);

        // "step" is how many words to advance the window each iteration.
        // Using (chunkWordCount - overlapWordCount) ensures the overlap region
        // is re-included at the start of the next chunk.
        var step = Math.Max(1, chunkWordCount - overlapWordCount);

        for (var i = 0; i < words.Length; i += step)
        {
            var length = Math.Min(chunkWordCount, words.Length - i);
            var chunk = string.Join(' ', words, i, length).Trim();

            if (!string.IsNullOrWhiteSpace(chunk))
            {
                yield return chunk;
            }

            // Exit early once the last word has been included in a chunk.
            if (i + length >= words.Length)
            {
                yield break;
            }
        }
    }

    /// <summary>
    /// Runs a Tesseract-compatible OCR CLI on a single image file and returns the
    /// recognised text.  Returns an empty string if the path is blank, the process
    /// fails to start, or the tool exits with a non-zero code.
    /// </summary>
    /// <param name="ocrCliPath">Full path to the OCR executable.</param>
    /// <param name="imagePath">Full path to the PNG image to analyse.</param>
    /// <returns>Normalised OCR text, or an empty string on any failure.</returns>
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

            // "stdout" tells Tesseract to write recognised text to standard output
            // instead of a file.  "--psm 6" selects "assume a single uniform block
            // of text", which works well for diagrams with captions.
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
            // Silently swallow exceptions (missing binary, permission errors, etc.)
            // so a broken OCR setup never crashes the indexing pipeline.
            return string.Empty;
        }
    }

    /// <summary>
    /// Collapses consecutive whitespace characters (spaces, tabs, newlines) in
    /// <paramref name="input"/> into a single space and trims leading/trailing whitespace.
    /// This keeps OCR output compact before it is stored as vector metadata.
    /// </summary>
    /// <param name="input">Raw text to normalise.</param>
    /// <returns>Normalised text, or an empty string if the input is whitespace-only.</returns>
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
                // Emit one space the first time we see whitespace, then suppress
                // any additional consecutive whitespace characters.
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