import logging

logger = logging.getLogger(__name__)

class DocumentProcessingLogger:
    def start_processing(self, doc_id, filename, domain):
        print(f"[LOG] Start processing: {doc_id} ({filename}) in {domain}")

    def log_upload_complete(self, size, content_type, storage_type, storage_path):
        print(f"[LOG] Upload complete: {size} bytes, {content_type}, {storage_type}")

    def log_extraction_start(self, suffix):
        print(f"[LOG] Extraction start: {suffix}")

    def log_extraction_complete(self, char_count, word_count):
        print(f"[LOG] Extraction complete: {char_count} chars, {word_count} words")

    def log_chunking_start(self, size, overlap):
        print(f"[LOG] Chunking start: size={size}, overlap={overlap}")

    def log_chunking_complete(self, count, avg_size):
        print(f"[LOG] Chunking complete: {count} chunks, avg size {avg_size}")

    def log_embedding_start(self, model, count):
        print(f"[LOG] Embedding start: model={model}, count={count}")

    def log_embedding_complete(self, count, dim):
        print(f"[LOG] Embedding complete: {count} vectors, dim={dim}")

    def log_persistence_start(self):
        print(f"[LOG] Persistence start")

    def log_persistence_complete(self, **kwargs):
        print(f"[LOG] Persistence complete: {kwargs}")

    def log_processing_complete(self, count, status):
        print(f"[LOG] Processing complete: {count} chunks, status={status}")

    def log_error(self, stage, error, **kwargs):
        print(f"[LOG] ERROR in {stage}: {error}")

    def log_warning(self, stage, message):
        print(f"[LOG] WARNING in {stage}: {message}")

    def log_vectorization_start(self, domain, count):
        print(f"[LOG] Vectorization start: {domain}, existing={count}")

    def log_vectorization_progress(self, stage, message):
        print(f"[LOG] Vectorization {stage}: {message}")

    def log_vectorization_complete(self, count, size_mb):
        print(f"[LOG] Vectorization complete: {count} vectors, {size_mb:.2f} MB")

_logger = DocumentProcessingLogger()

def get_document_processing_logger():
    return _logger
