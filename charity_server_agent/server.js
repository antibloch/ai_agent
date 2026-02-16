const express = require("express");

const app = express();
const PORT = 3000;

/**
 * ----------------------------
 * Dummy "DB" (in-memory)
 * ----------------------------
 * Shaped from your collections/fields:
 * - charityorganizations: core profile + address + country availability, etc. :contentReference[oaicite:2]{index=2}
 * - charityrankings: donationAmount, impactLife, donors :contentReference[oaicite:3]{index=3}
 * - charityproducts: name, pricePerUnit, totalDonated, category, charity :contentReference[oaicite:4]{index=4}
 * - charityblogs: title, description, file, charity :contentReference[oaicite:5]{index=5}
 */

// Simple deterministic IDs (string instead of ObjectId for dummy mode)
const ORG_1 = "org_001";
const ORG_2 = "org_002";

const charityorganizations = [
  {
    _id: ORG_1,
    name: "Helping Hands Foundation",
    description: "Food & healthcare support for underserved communities.",
    registrationNumber: "REG-PAK-001",
    email: "helpinghands@yopmail.com",
    phone: "+923001112223",
    website: "https://helpinghands.example",
    logo: "/uploads/logos/helpinghands.png",
    partOfGiver: true,
    verificationStatus: "Approved",
    isDeleted: false,
    isSuspended: false,
    user: "user_101",
    address: {
      street: "12 Service Rd",
      city: "Karachi",
      state: "Sindh",
      country: "Pakistan",
      countryCode: "PK",
      postalCode: "74000",
      latitude: 24.8607,
      longitude: 67.0011,
    },
    CountryAvailability: [
      { _id: "ca_1", country: "Pakistan", countryCode: "PK" },
      { _id: "ca_2", country: "UAE", countryCode: "AE" },
    ],
    documents: {
      registrationCertificate: {
        name: "Registration Certificate",
        url: "uploads\\verification_documents\\regcert.pdf",
        expiryDate: null,
        verified: "verified",
      },
      taxExemptionCertificate: {
        name: "Tax Exemption Certificate",
        url: "uploads\\verification_documents\\taxexempt.pdf",
        expiryDate: null,
        verified: "verified",
      },
      annualReport: {
        name: "Annual Report",
        url: "uploads\\verification_documents\\annualreport.pdf",
        expiryDate: null,
        verified: "verified",
        year: 2026,
      },
      governmentApproval: {
        name: "Government Approval",
        url: "uploads\\verification_documents\\govt.pdf",
        expiryDate: null,
        verified: "verified",
      },
    },
    paymentCustomerId: "cus_dummy_001",
    walletUid: "wallet_dummy_001",
    createdAt: "2026-01-10T10:00:00.000Z",
    updatedAt: "2026-02-01T10:00:00.000Z",
    __v: 0,
  },
  {
    _id: ORG_2,
    name: "Bright Futures Trust",
    description: "Education kits and scholarships for students.",
    registrationNumber: "REG-PAK-002",
    email: "brightfutures@yopmail.com",
    phone: "+923004445556",
    website: "https://brightfutures.example",
    logo: "/uploads/logos/brightfutures.png",
    partOfGiver: true,
    verificationStatus: "Approved",
    isDeleted: false,
    isSuspended: false,
    user: "user_202",
    address: {
      street: "88 University Ave",
      city: "Lahore",
      state: "Punjab",
      country: "Pakistan",
      countryCode: "PK",
      postalCode: "54000",
      latitude: 31.5204,
      longitude: 74.3587,
    },
    CountryAvailability: [{ _id: "ca_3", country: "Pakistan", countryCode: "PK" }],
    documents: {
      registrationCertificate: {
        name: "Registration Certificate",
        url: "uploads\\verification_documents\\regcert2.pdf",
        expiryDate: null,
        verified: "verified",
      },
      taxExemptionCertificate: {
        name: "Tax Exemption Certificate",
        url: "uploads\\verification_documents\\taxexempt2.pdf",
        expiryDate: null,
        verified: "verified",
      },
      annualReport: {
        name: "Annual Report",
        url: "uploads\\verification_documents\\annualreport2.pdf",
        expiryDate: null,
        verified: "verified",
        year: 2026,
      },
      governmentApproval: {
        name: "Government Approval",
        url: "uploads\\verification_documents\\govt2.pdf",
        expiryDate: null,
        verified: "verified",
      },
    },
    paymentCustomerId: "cus_dummy_002",
    walletUid: "wallet_dummy_002",
    createdAt: "2026-01-12T10:00:00.000Z",
    updatedAt: "2026-02-02T10:00:00.000Z",
    __v: 0,
  },
];

const charityrankings = [
  {
    _id: "rank_1",
    charityId: ORG_1,
    userId: "user_101",
    country: "Pakistan",
    donationAmount: 250000,
    impactLife: 1200,
    donors: ["don_1", "don_2", "don_3", "don_4"],
    createdAt: "2026-01-20T10:00:00.000Z",
    updatedAt: "2026-02-01T10:00:00.000Z",
    __v: 0,
  },
  {
    _id: "rank_2",
    charityId: ORG_2,
    userId: "user_202",
    country: "Pakistan",
    donationAmount: 180000,
    impactLife: 800,
    donors: ["don_5", "don_6"],
    createdAt: "2026-01-21T10:00:00.000Z",
    updatedAt: "2026-02-01T10:00:00.000Z",
    __v: 0,
  },
];

const charityproducts = [
  {
    _id: "prod_1",
    charity: ORG_1,
    name: "Food Pack",
    description: "Monthly ration pack for a family.",
    pricePerUnit: 5000,
    totalDonated: 43,
    category: "cat_food",
    partner: "partner_1",
    parent: "parent_1",
    isFeatured: true,
    isActive: true,
    isDeleted: false,
    status: "approved",
    images: [{ _id: "img_1", url: "/uploads/products/foodpack.png", isPrimary: true }],
    createdAt: "2026-01-15T10:00:00.000Z",
    updatedAt: "2026-02-01T10:00:00.000Z",
    __v: 0,
  },
  {
    _id: "prod_2",
    charity: ORG_1,
    name: "Clinic Voucher",
    description: "Basic checkup voucher for 1 person.",
    pricePerUnit: 2000,
    totalDonated: 75,
    category: "cat_health",
    partner: "partner_1",
    parent: "parent_1",
    isFeatured: false,
    isActive: true,
    isDeleted: false,
    status: "approved",
    images: [{ _id: "img_2", url: "/uploads/products/voucher.png", isPrimary: true }],
    createdAt: "2026-01-16T10:00:00.000Z",
    updatedAt: "2026-02-01T10:00:00.000Z",
    __v: 0,
  },
  {
    _id: "prod_3",
    charity: ORG_2,
    name: "School Kit",
    description: "Backpack + stationery for 1 student.",
    pricePerUnit: 3500,
    totalDonated: 61,
    category: "cat_edu",
    partner: "partner_2",
    parent: "parent_2",
    isFeatured: true,
    isActive: true,
    isDeleted: false,
    status: "approved",
    images: [{ _id: "img_3", url: "/uploads/products/schoolkit.png", isPrimary: true }],
    createdAt: "2026-01-18T10:00:00.000Z",
    updatedAt: "2026-02-01T10:00:00.000Z",
    __v: 0,
  },
];

const charityblogs = [
  {
    _id: "blog_1",
    charity: ORG_1,
    title: "Winter Relief Drive",
    description: "<p>We delivered food packs across 3 districts.</p>",
    hashtags: ["#relief", "#winter"],
    file: "/uploads/blogs/winterrelief.png",
    status: "draft",
    isDeleted: false,
    createdAt: "2026-01-25T10:00:00.000Z",
    updatedAt: "2026-02-01T10:00:00.000Z",
    __v: 0,
  },
  {
    _id: "blog_2",
    charity: ORG_2,
    title: "Scholarships Awarded",
    description: "<p>We awarded 50 scholarships this quarter.</p>",
    hashtags: ["#education", "#scholarship"],
    file: "/uploads/blogs/scholarships.png",
    status: "draft",
    isDeleted: false,
    createdAt: "2026-01-26T10:00:00.000Z",
    updatedAt: "2026-02-01T10:00:00.000Z",
    __v: 0,
  },
];

/**
 * Helpers
 */
function byIdOrg(orgId) {
  return charityorganizations.find((c) => c._id === orgId);
}

function orgName(orgId) {
  const org = byIdOrg(orgId);
  return org ? org.name : `UNKNOWN(${orgId})`;
}

function stableEnvelope({ tool, query, data, warnings = [] }) {
  return {
    ok: true,
    tool,
    query,
    data,
    meta: {
      dummy: true,
      warnings,
      timestamp: new Date().toISOString(),
    },
  };
}

/**
 * ----------------------------
 * "Tools" implemented over dummy DB
 * Names based on your tools.md :contentReference[oaicite:6]{index=6}
 * ----------------------------
 */

// charity_donor_count(List[org.name], List[ranking.charityId], List[ranking.donors])
function charity_donor_count() {
  return charityrankings.map((r) => ({
    charityName: orgName(r.charityId),
    donorCount: Array.isArray(r.donors) ? r.donors.length : 0,
  }));
}

// charity_impactlife(List[org.name], List[ranking.charityId])
function charity_impactlife() {
  return charityrankings.map((r) => ({
    charityName: orgName(r.charityId),
    impactLife: r.impactLife ?? 0,
  }));
}

// charity_donor_amount(List[org.name], List[ranking.charityId])
function charity_donor_amount() {
  return charityrankings.map((r) => ({
    charityName: orgName(r.charityId),
    donationAmount: r.donationAmount ?? 0,
  }));
}

// charity_total_donation(List[products.charity], List[products.name], List[products.totalDonated])
function charity_total_donation() {
  const out = {};
  for (const p of charityproducts) {
    const name = orgName(p.charity);
    if (!out[name]) out[name] = {};
    out[name][p.name] = p.totalDonated ?? 0;
  }
  // Return as list for agent-friendliness
  return Object.entries(out).map(([charityName, products]) => ({
    charityName,
    products, // { productName: totalDonated }
  }));
}

// charity_items_category(List[products.charity], List[products.category])
function charity_items_category() {
  const out = {};
  for (const p of charityproducts) {
    const name = orgName(p.charity);
    if (!out[name]) out[name] = new Set();
    out[name].add(p.category);
  }
  return Object.entries(out).map(([charityName, categoriesSet]) => ({
    charityName,
    productCategories: Array.from(categoriesSet),
  }));
}

// charity_product_price_description(charityproducts)
function charity_product_price_description() {
  const out = {};
  for (const p of charityproducts) {
    const name = orgName(p.charity);
    if (!out[name]) out[name] = [];
    out[name].push({
      productName: p.name,
      pricePerUnit: p.pricePerUnit,
      description: p.description,
    });
  }
  return Object.entries(out).map(([charityName, products]) => ({
    charityName,
    products, // list of { productName, pricePerUnit, description }
  }));
}

// charity_blogs(charityblogs)
function charity_blogs() {
  const out = {};
  for (const b of charityblogs) {
    const name = orgName(b.charity);
    if (!out[name]) out[name] = [];
    out[name].push({
      title: b.title,
      description: b.description, // HTML string (as in your samples)
      file: b.file,
      hashtags: b.hashtags ?? [],
    });
  }
  return Object.entries(out).map(([charityName, blogs]) => ({
    charityName,
    blogs,
  }));
}

// charity_address(charityorganizations)
function charity_address() {
  return charityorganizations.map((c) => ({
    charityName: c.name,
    address: {
      street: c.address?.street ?? null,
      city: c.address?.city ?? null,
      state: c.address?.state ?? null,
      country: c.address?.country ?? null,
      countryCode: c.address?.countryCode ?? null,
      postalCode: c.address?.postalCode ?? null,
    },
  }));
}

// charity_country_availability(charityorganizations)
function charity_country_availability() {
  return charityorganizations.map((c) => ({
    charityName: c.name,
    CountryAvailability: (c.CountryAvailability ?? []).map((x) => ({
      country: x.country,
      countryCode: x.countryCode,
    })),
  }));
}

// charity_contact_info(charityorganizations)  (note the typo in your tools.md, we support both)
function charity_contact_info() {
  return charityorganizations.map((c) => ({
    charityName: c.name,
    contact: {
      email: c.email,
      phone: c.phone,
      website: c.website,
    },
  }));
}

/**
 * Router mapping:
 * - Accepts q=tool_name (and a few aliases)
 * - Returns stable JSON envelope
 */
function handleToolQuery(rawQ) {
  const q = String(rawQ || "").trim();
  const norm = q.toLowerCase();

  // Allow a couple of practical aliases so the agent can be sloppy
  const toolMap = [
    { keys: ["charity_donor_count", "donor_count", "donors_count"], fn: charity_donor_count },
    { keys: ["charity_impactlife", "impactlife", "impact_life"], fn: charity_impactlife },
    { keys: ["charity_donor_amount", "donor_amount", "donation_amount"], fn: charity_donor_amount },
    { keys: ["charity_total_donation", "total_donation", "product_total_donation"], fn: charity_total_donation },
    { keys: ["charity_items_category", "items_category", "product_categories"], fn: charity_items_category },
    { keys: ["charity_product_price_description", "product_price_description", "products_info"], fn: charity_product_price_description },
    { keys: ["charity_blogs", "blogs"], fn: charity_blogs },
    { keys: ["charity_address", "address"], fn: charity_address },
    { keys: ["charity_country_availability", "country_availability"], fn: charity_country_availability },
    { keys: ["charity_contact_info", "charity_contact_info", "contact_info"], fn: charity_contact_info },
  ];

  for (const entry of toolMap) {
    if (entry.keys.some((k) => norm === k)) {
      return stableEnvelope({
        tool: entry.keys[0],
        query: q,
        data: entry.fn(),
      });
    }
  }

  // Generic fallback “stats” (useful for queries like "total charities")
  // Keeps agent from failing hard.
  const fallbackData = {
    totals: {
      charityorganizations: charityorganizations.length,
      charityrankings: charityrankings.length,
      charityproducts: charityproducts.length,
      charityblogs: charityblogs.length,
    },
    charities: charityorganizations.map((c) => ({
      _id: c._id,
      name: c.name,
      verificationStatus: c.verificationStatus,
      country: c.address?.country ?? null,
      countryCode: c.address?.countryCode ?? null,
    })),
  };

  return {
    ok: false,
    tool: "unknown_tool",
    query: q,
    error: `Unknown tool query: "${q}". Try one of: ${toolMap.map((x) => x.keys[0]).join(", ")}`,
    data: fallbackData,
    meta: { dummy: true, timestamp: new Date().toISOString() },
  };
}

/**
 * Single GET endpoint (your required single pathway)
 * http://localhost:3000/api/stats?q=charity_donor_count
 */
app.get("/api/stats", (req, res) => {
  const q = req.query.q || "";
  const payload = handleToolQuery(q);

  // Always return JSON, always 200 (tool callers often prefer a JSON error to exceptions)
  res.setHeader("Content-Type", "application/json");
  res.status(200).json(payload);
});

app.listen(PORT, () => {
  console.log(`Node API running on http://localhost:${PORT}`);
  console.log(`Try: http://localhost:${PORT}/api/stats?q=charity_donor_count`);
});
