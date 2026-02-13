from pymongo import MongoClient
from bson import json_util
from collections import defaultdict
import json

MONGO_URI = "mongodb+srv://giverr:giverr123@giverr.qy0czq5.mongodb.net/?retryWrites=true&w=majority&appName=giverr"

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
client.admin.command("ping")
print("✅ Connected")

db = client["test"]

collections_to_explore = [
    "charityorganizations",
    "charities",
    "charityblogs",
    "charityproducts",
    "charityrankings",
]

def summarize_schema(sample_docs):
    field_types = defaultdict(set)

    def walk(doc, prefix=""):
        if isinstance(doc, dict):
            for k, v in doc.items():
                key = f"{prefix}.{k}" if prefix else k
                field_types[key].add(type(v).__name__)
                walk(v, key)
        elif isinstance(doc, list):
            for item in doc[:3]:  # limit depth
                walk(item, prefix)

    for doc in sample_docs:
        walk(doc)

    return {k: list(v) for k, v in sorted(field_types.items())}


for coll_name in collections_to_explore:
    print("\n" + "=" * 80)
    print(f"📦 COLLECTION: {coll_name}")

    if coll_name not in db.list_collection_names():
        print("❌ Collection not found.")
        continue

    collection = db[coll_name]

    total = collection.count_documents({})
    print("Total documents:", total)

    sample_docs = list(collection.find({}).limit(2))
    print("\n--- SAMPLE DOCUMENTS ---")
    print(json_util.dumps(sample_docs, indent=2))


    print("\n--- INFERRED FIELD SUMMARY ---")
    schema_summary = summarize_schema(sample_docs)
    for field, types in schema_summary.items():
        print(f"{field}: {types}")
