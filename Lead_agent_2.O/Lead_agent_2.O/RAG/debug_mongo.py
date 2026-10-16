import os
from app.core.database import init_db
from app.models.mongo_models import KnowledgeDocument, DocumentStatus

def debug_mongo():
    print("Initializing DB...")
    init_db()
    
    print("\nListing all KnowledgeDocuments:")
    docs = KnowledgeDocument.objects()
    print(f"Total documents found: {len(docs)}")
    
    for doc in docs:
        print(f" - DocID: {doc.doc_id}")
        print(f"   Original Name: {doc.original_name}")
        print(f"   Domain: {doc.domain}")
        print(f"   Status: {doc.status}")
        print(f"   Embedding Status: {doc.embedding_status}")
        print("-" * 20)

    print("\nChecking for Active documents in domain_3:")
    active_docs = KnowledgeDocument.objects(
        domain="domain_3",
        status=DocumentStatus.ACTIVE
    )
    print(f"Active docs count: {len(active_docs)}")
    for doc in active_docs:
        print(f" -> {doc.doc_id}")

if __name__ == "__main__":
    debug_mongo()
