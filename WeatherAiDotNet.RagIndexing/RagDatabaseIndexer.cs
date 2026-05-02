namespace WeatherAiDotNet.RagIndexing;

public static class RagDatabaseIndexer
{
    public static async Task<RagIndexingSummary> IndexAsync(
        string pdfFolderPath,
        bool includeImages,
        Func<string, string> readPdfText,
        Func<string, IEnumerable<string>> chunkText,
        Func<string, IReadOnlyList<RagImageItem>> extractImageItems,
        Func<string, Task<float[]>> embedAsync,
        Func<string, string, bool> isDocumentUpToDate,
        Action<string, string, IReadOnlyList<RagStoredVector>> replaceDocumentVectors,
        Action<string> logInfo,
        Action<string, Exception>? logWarning)
    {
        logInfo("Reading PDFs...");

        var pdfFiles = Directory
            .EnumerateFiles(pdfFolderPath, "*.pdf", SearchOption.AllDirectories)
            .OrderBy(static x => x, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (pdfFiles.Count == 0)
        {
            logInfo($"No PDF files found under: {pdfFolderPath}");
            return new RagIndexingSummary(0, 0, 0, 0);
        }

        logInfo($"Found {pdfFiles.Count} PDF files.");

        var reindexedDocuments = 0;
        var skippedDocuments = 0;
        var totalIndexedChunks = 0;
        var totalImageItems = 0;

        foreach (var file in pdfFiles)
        {
            var relativePdfPath = Path.GetRelativePath(pdfFolderPath, file);
            var fingerprint = BuildDocumentFingerprint(file);

            if (isDocumentUpToDate(relativePdfPath, fingerprint))
            {
                skippedDocuments++;
                logInfo($"Skipping unchanged PDF: {relativePdfPath}");
                continue;
            }

            logInfo($"Indexing PDF: {relativePdfPath}");

            string pdfText;
            try
            {
                pdfText = readPdfText(file);
            }
            catch (Exception ex)
            {
                logWarning?.Invoke("Skipping unreadable PDF.", ex);
                continue;
            }

            var chunkIndex = 0;
            var fileVectors = new List<RagStoredVector>();

            var chunks = chunkText(pdfText).ToList();
            foreach (var chunk in chunks)
            {
                var vector = await embedAsync(chunk);
                fileVectors.Add(new RagStoredVector(chunkIndex++, chunk, vector));
            }

            if (includeImages)
            {
                var imageItems = extractImageItems(file);
                totalImageItems += imageItems.Count;

                foreach (var imageItem in imageItems)
                {
                    var vector = await embedAsync(imageItem.IndexText);
                    fileVectors.Add(new RagStoredVector(chunkIndex++, imageItem.IndexText, vector));
                }
            }

            if (fileVectors.Count == 0)
            {
                continue;
            }

            replaceDocumentVectors(relativePdfPath, fingerprint, fileVectors);
            reindexedDocuments++;
            totalIndexedChunks += fileVectors.Count;
        }

        return new RagIndexingSummary(reindexedDocuments, skippedDocuments, totalIndexedChunks, totalImageItems);
    }

    private static string BuildDocumentFingerprint(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        return $"{fileInfo.Length}:{fileInfo.LastWriteTimeUtc.Ticks}";
    }
}
