from pymongo import MongoClient

MONGO_URI = "mongodb+srv://giverr:giverr123@giverr.qy0czq5.mongodb.net/?retryWrites=true&w=majority&appName=giverr"

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
db = client["test"]
charities = db["charityorganizations"]

donor_visible_filter = {
    "verificationStatus": "Approved",
    "isDeleted": False,
    "isSuspended": False,

    "documents.registrationCertificate.verified": "verified",
    "documents.taxExemptionCertificate.verified": "verified",
    "documents.annualReport.verified": "verified",
    "documents.governmentApproval.verified": "verified",
}

# Return “associated info” but exclude sensitive payment fields (per schema note).
projection = {
    "paymentCustomerId": 0,
    "walletUid": 0,
    "defaultPaymentMethod": 0,
}

docs = list(charities.find(donor_visible_filter, projection).sort("createdAt", -1))
print("Donor-visible charities:", len(docs))

for d in docs[:10]:
    print(d.get("name"), d.get("email"), d.get("address", {}).get("countryCode"))


print("Total charities:", charities.count_documents({}))

all_charities = list(charities.find({}))
print("Total:", len(all_charities))

for c in all_charities:
    print("="*60)
    print(c)
