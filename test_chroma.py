import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "test_collection"
PERSIST_DIRECTORY = ".chroma"

client = chromadb.PersistentClient(path=PERSIST_DIRECTORY, settings=Settings(anonymized_telemetry=False))

# Clean up any existing collection
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"No existing collection to delete: {e}")

# Create collection
collection = client.create_collection(name=COLLECTION_NAME)
print(f"Created collection: {COLLECTION_NAME}")

# Add some dummy data
collection.add(
    embeddings=[[0.1, 0.2, 0.3]],
    metadatas=[{"source": "test.pdf", "page": 1}],
    documents=["test text"],
    ids=["id1"]
)
print(f"After adding, count: {collection.count()}")

# Delete collection
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Deleted collection after add: {COLLECTION_NAME}")
except Exception as e:
    print(f"Error deleting: {e}")

# Try to get the collection (should raise)
try:
    coll = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection still exists? count: {coll.count()}")
except Exception as e:
    print(f"Collection does not exist after delete: {e}")

# Create again
collection2 = client.create_collection(name=COLLECTION_NAME)
print(f"Created collection again: {COLLECTION_NAME}")
print(f"Count: {collection2.count()}")