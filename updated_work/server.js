const express = require("express");

const app = express();
const PORT = 3000;

// ───────────────────────────────────────────────
//  Constants
// ───────────────────────────────────────────────

// Authentication token (matches Postman: x-auth-token)
const AUTH_TOKEN = "charity-demo-token-2026";

// Dummy DB IDs
const ORG_1 = "org_001";
const ORG_2 = "org_002";

// ───────────────────────────────────────────────
// Dummy Database Collections (fully expanded)
// ───────────────────────────────────────────────

const charityorganizations = [
  {
    _id: ORG_1,
    name: "Helping Hands Foundation",
    description: "Food & healthcare support for underserved communities.",
    registrationNumber: "REG-PAK-001",
    email: "helpinghands@yopmail.com",
    phone: "+923001112223",
    website: "https://helpinghandfound.org/pakistan/",
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
      registrationCertificate: { name: "Registration Certificate", url: "uploads\\verification_documents\\regcert.pdf", expiryDate: null, verified: "verified" },
      taxExemptionCertificate: { name: "Tax Exemption Certificate", url: "uploads\\verification_documents\\taxexempt.pdf", expiryDate: null, verified: "verified" },
      annualReport: { name: "Annual Report", url: "uploads\\verification_documents\\annualreport.pdf", expiryDate: null, verified: "verified", year: 2026 },
      governmentApproval: { name: "Government Approval", url: "uploads\\verification_documents\\govt.pdf", expiryDate: null, verified: "verified" },
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
    website: "https://www.aap.org/en/practice-management/bright-futures/",
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
      registrationCertificate: { name: "Registration Certificate", url: "uploads\\verification_documents\\regcert2.pdf", expiryDate: null, verified: "verified" },
      taxExemptionCertificate: { name: "Tax Exemption Certificate", url: "uploads\\verification_documents\\taxexempt2.pdf", expiryDate: null, verified: "verified" },
      annualReport: { name: "Annual Report", url: "uploads\\verification_documents\\annualreport2.pdf", expiryDate: null, verified: "verified", year: 2026 },
      governmentApproval: { name: "Government Approval", url: "uploads\\verification_documents\\govt2.pdf", expiryDate: null, verified: "verified" },
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
    donors: [
      { donorId: "don_1", totalAmount: 100000, impactLife: 500 },
      { donorId: "don_2", totalAmount: 50000, impactLife: 250 },
      { donorId: "don_3", totalAmount: 50000, impactLife: 250 },
      { donorId: "don_4", totalAmount: 50000, impactLife: 250 },
    ],
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
    donors: [
      { donorId: "don_5", totalAmount: 100000, impactLife: 400 },
      { donorId: "don_6", totalAmount: 80000, impactLife: 400 },
    ],
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

// ───────────────────────────────────────────────
// Helpers
// ───────────────────────────────────────────────

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

function paginate(array, page = 1, limit = 10) {
  const start = (page - 1) * limit;
  const end = start + limit;
  return {
    items: array.slice(start, end),
    pagination: {
      total: array.length,
      page: Number(page),
      limit: Number(limit),
      totalPages: Math.ceil(array.length / limit),
      hasNext: end < array.length,
      hasPrev: start > 0,
    },
  };
}

function getCharityIdFromToken(token) {
  if (token !== AUTH_TOKEN) return null;
  return ORG_1; // dummy: always returns ORG_1 (change to ORG_2 to test other charity)
}

// ───────────────────────────────────────────────
// Tool Functions (original)
// ───────────────────────────────────────────────

function charity_donor_count() {
  return charityrankings.map((r) => ({
    charityName: orgName(r.charityId),
    donorCount: Array.isArray(r.donors) ? r.donors.length : 0,
  }));
}

function charity_impactlife() {
  return charityrankings.map((r) => ({
    charityName: orgName(r.charityId),
    impactLife: r.impactLife ?? 0,
  }));
}

function charity_donor_amount() {
  return charityrankings.map((r) => ({
    charityName: orgName(r.charityId),
    donationAmount: r.donationAmount ?? 0,
  }));
}

function charity_total_donation() {
  const out = {};
  for (const p of charityproducts) {
    const name = orgName(p.charity);
    if (!out[name]) out[name] = {};
    out[name][p.name] = p.totalDonated ?? 0;
  }
  return Object.entries(out).map(([charityName, products]) => ({
    charityName,
    products,
  }));
}

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
    products,
  }));
}

function charity_blogs() {
  const out = {};
  for (const b of charityblogs) {
    const name = orgName(b.charity);
    if (!out[name]) out[name] = [];
    out[name].push({
      title: b.title,
      description: b.description,
      file: b.file,
      hashtags: b.hashtags ?? [],
    });
  }
  return Object.entries(out).map(([charityName, blogs]) => ({
    charityName,
    blogs,
  }));
}

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

function charity_country_availability() {
  return charityorganizations.map((c) => ({
    charityName: c.name,
    CountryAvailability: (c.CountryAvailability ?? []).map((x) => ({
      country: x.country,
      countryCode: x.countryCode,
    })),
  }));
}

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

// ───────────────────────────────────────────────
// Tool Query Handler (original)
// ───────────────────────────────────────────────

function handleToolQuery(rawQ) {
  const q = String(rawQ || "").trim();
  const norm = q.toLowerCase();

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
    { keys: ["charity_contact_info", "contact_info"], fn: charity_contact_info },
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

  // Fallback
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

// ───────────────────────────────────────────────
// Routes
// ───────────────────────────────────────────────

// Original tool endpoint
/**
 * Legacy / tool-based stats endpoint (used by agent / internal tools)
 * @route   GET /api/stats
 * @access  Public
 * @query   {string} q              The tool name / alias to execute
 *                                  Examples:
 *                                  - charity_donor_count
 *                                  - charity_impactlife
 *                                  - charity_total_donation
 *                                  - charity_blogs
 *                                  - charity_address
 *                                  - etc.
 * @returns {200}                   Always returns 200 even on error (tool-friendly)
 *          Success case:
 *          {
 *            "ok": true,
 *            "tool": string,
 *            "query": string,
 *            "data": array | object,
 *            "meta": { dummy: true, warnings: [], timestamp: string }
 *          }
 *
 *          Error / unknown tool case:
 *          {
 *            "ok": false,
 *            "tool": "unknown_tool",
 *            "query": string,
 *            "error": string,
 *            "data": { totals: {...}, charities: [...] },
 *            "meta": { dummy: true, timestamp: string }
 *          }
 *
 * @example
 *   GET /api/stats?q=charity_donor_count
 *   → returns list of { charityName, donorCount }
 */
app.get("/api/stats", (req, res) => {
  const q = req.query.q || "";
  const payload = handleToolQuery(q);
  res.setHeader("Content-Type", "application/json");
  res.status(200).json(payload);
});


// 1. Search Charity (public)
/**
 * Search approved charities by name or email
 * @route   GET /api/v1/charity_organization/search
 * @access  Public
 * @query   {string} search          Required. Search term (charity name or email)
 * @returns {200}                    Success response
 *          {
 *            "success": true,
 *            "message": "Charity search completed successfully",
 *            "data": {
 *              "searchQuery": string,
 *              "totalResults": number,
 *              "charities": [
 *                { "_id": string, "name": string, "email": string, "logo": string|null, "address": object, "verificationStatus": string }
 *              ]
 *            }
 *          }
 * @returns {400}                    Missing search query
 *          { "success": false, "message": "Search query is required" }
 */
app.get("/api/v1/charity_organization/search", (req, res) => {
  const search = (req.query.search || "").trim().toLowerCase();
  if (!search) {
    return res.status(400).json({ success: false, message: "Search query is required" });
  }

  const filtered = charityorganizations.filter(
    (c) =>
      c.verificationStatus === "Approved" &&
      !c.isDeleted &&
      !c.isSuspended &&
      (c.name.toLowerCase().includes(search) || c.email.toLowerCase().includes(search))
  );

  const charities = filtered.map((c) => ({
    _id: c._id,
    name: c.name,
    email: c.email,
    logo: c.logo,
    address: c.address,
    verificationStatus: c.verificationStatus,
  }));

  res.json({
    success: true,
    message: "Charity search completed successfully",
    data: { searchQuery: search, totalResults: charities.length, charities },
  });
});


// 2. Get Charity Profile By ID (public)
/**
 * Get detailed profile of a single charity
 * @route   GET /api/v1/charity_organization/get-charity-profile/:charityId
 * @access  Public
 * @param   {string} charityId       MongoDB-style ID of the charity (path param)
 * @returns {200}                    Success
 *          {
 *            "success": true,
 *            "message": "Charity profile fetched successfully",
 *            "charity": {
 *              "_id": string,
 *              "name": string,
 *              "email": string,
 *              "phone": string,
 *              "logo": string|null,
 *              "address": object,
 *              "registrationNumber": string,
 *              "verificationStatus": string,
 *              "description": string,
 *              "website": string|null,
 *              "createdAt": string (ISO),
 *              "updatedAt": string (ISO)
 *            }
 *          }
 * @returns {404}                    Charity not found or not approved
 *          { "success": false, "message": "Charity not found or not approved" }
 */
app.get("/api/v1/charity_organization/get-charity-profile/:charityId", (req, res) => {
  const charity = byIdOrg(req.params.charityId);
  if (!charity || charity.verificationStatus !== "Approved" || charity.isDeleted || charity.isSuspended) {
    return res.status(404).json({ success: false, message: "Charity not found or not approved" });
  }

  res.json({
    success: true,
    message: "Charity profile fetched successfully",
    charity: {
      _id: charity._id,
      name: charity.name,
      email: charity.email,
      phone: charity.phone,
      logo: charity.logo,
      address: charity.address,
      registrationNumber: charity.registrationNumber,
      verificationStatus: charity.verificationStatus,
      description: charity.description,
      website: charity.website,
      createdAt: charity.createdAt,
      updatedAt: charity.updatedAt,
    },
  });
});

// 3. Get Charity Products (authenticated)
/**
 * Get paginated list of products belonging to the authenticated charity
 * @route   GET /api/v1/products/get-charity-products
 * @access  Private (charity organization)
 * @header  {string} x-auth-token    Authentication token
 * @query   {number}  [page=1]
 * @query   {number}  [limit=10]
 * @query   {string}  [isActive=true|false]
 * @query   {string}  [isDeleted=true|false]
 * @query   {string}  [status=approved|pending|rejected]
 * @query   {string}  [productId]         Filter by parent product ID
 * @query   {number}  [minPrice]
 * @query   {number}  [maxPrice]
 * @query   {string}  [startDate]         ISO date
 * @query   {string}  [endDate]           ISO date
 * @query   {string}  [category]          comma-separated category IDs
 * @query   {string}  [search]
 * @query   {string}  [sort]              e.g. -createdAt, price, name
 * @returns {200}                         Success
 *          {
 *            "success": true,
 *            "message": "Charity products fetched successfully",
 *            "data": {
 *              "products": [
 *                {
 *                  "_id": string,
 *                  "name": string,
 *                  "description": string,
 *                  "pricePerUnit": number,
 *                  "isActive": boolean,
 *                  "status": string,
 *                  "charity": { "_id": string, "name": string, "logo": string|null },
 *                  "parent": { ... } | null,
 *                  "createdAt": string,
 *                  "updatedAt": string
 *                }
 *              ],
 *              "pagination": { total, page, limit, totalPages, hasNext, hasPrev }
 *            }
 *          }
 * @returns {401}                         Unauthorized
 *          { "success": false, "message": "Unauthorized" }
 */
app.get("/api/v1/products/get-charity-products", (req, res) => {
  const token = req.headers["x-auth-token"];
  const charityId = getCharityIdFromToken(token);
  if (!charityId) return res.status(401).json({ success: false, message: "Unauthorized" });

  let products = charityproducts.filter((p) => p.charity === charityId);

  // Filters (matching Postman)
  if (req.query.isActive !== undefined) products = products.filter((p) => p.isActive === (req.query.isActive === "true"));
  if (req.query.isDeleted !== undefined) products = products.filter((p) => p.isDeleted === (req.query.isDeleted === "true"));
  if (req.query.status) products = products.filter((p) => p.status === req.query.status);
  if (req.query.productId) products = products.filter((p) => p.parent === req.query.productId);
  if (req.query.minPrice) products = products.filter((p) => p.pricePerUnit >= Number(req.query.minPrice));
  if (req.query.maxPrice) products = products.filter((p) => p.pricePerUnit <= Number(req.query.maxPrice));
  if (req.query.startDate) products = products.filter((p) => new Date(p.createdAt) >= new Date(req.query.startDate));
  if (req.query.endDate) products = products.filter((p) => new Date(p.createdAt) <= new Date(req.query.endDate));
  if (req.query.category) {
    const cats = req.query.category.split(",");
    products = products.filter((p) => cats.includes(p.category));
  }
  if (req.query.search) {
    const s = req.query.search.toLowerCase();
    products = products.filter((p) => p.name.toLowerCase().includes(s) || p.description.toLowerCase().includes(s));
  }

  // Sorting
  let sortField = req.query.sort || "createdAt";
  let sortDir = 1;
  if (sortField.startsWith("-")) {
    sortDir = -1;
    sortField = sortField.slice(1);
  }
  products.sort((a, b) => sortDir * (a[sortField] > b[sortField] ? 1 : a[sortField] < b[sortField] ? -1 : 0));

  const page = Number(req.query.page) || 1;
  const limit = Number(req.query.limit) || 10;
  const { items, pagination } = paginate(products, page, limit);

  const enriched = items.map((p) => ({
    _id: p._id,
    name: p.name,
    description: p.description,
    pricePerUnit: p.pricePerUnit,
    isActive: p.isActive,
    status: p.status,
    charity: { _id: charityId, name: orgName(charityId), logo: byIdOrg(charityId)?.logo },
    parent: p.parent ? { _id: p.parent, name: "Parent Product", category: { _id: p.category, name: "Category Name", color: "#FF5733" }, partner: { _id: p.partner, name: "Partner Name", verificationStatus: "Approved" } } : null,
    createdAt: p.createdAt,
    updatedAt: p.updatedAt,
  }));

  res.json({
    success: true,
    message: "Charity products fetched successfully",
    data: { products: enriched, pagination },
  });
});

// 4. Get Charity Blogs (authenticated)
/**
 * Get paginated list of blogs belonging to the authenticated charity
 * @route   GET /api/v1/charity_organization/blogs
 * @access  Private (charity organization)
 * @header  {string} x-auth-token
 * @query   {number}  [page=1]
 * @query   {number}  [limit=10]          max 100
 * @query   {string}  [search]            title, description or hashtags
 * @query   {string}  [sortBy=createdAt]  createdAt|updatedAt|title|status
 * @query   {string}  [order=desc]        asc|desc
 * @returns {200}
 *          {
 *            "success": true,
 *            "message": "Blogs fetched successfully",
 *            "blogs": [
 *              {
 *                "_id": string,
 *                "charity": string,
 *                "title": string,
 *                "description": string,
 *                "hashtags": string[],
 *                "file": string|null,
 *                "status": string,
 *                "isDeleted": boolean,
 *                "createdAt": string,
 *                "updatedAt": string
 *              }
 *            ],
 *            "pagination": { ... + sortBy, order, search }
 *          }
 * @returns {401}                         Unauthorized
 */
app.get("/api/v1/charity_organization/blogs", (req, res) => {
  const token = req.headers["x-auth-token"];
  const charityId = getCharityIdFromToken(token);
  if (!charityId) return res.status(401).json({ success: false, message: "Unauthorized" });

  let blogs = charityblogs.filter((b) => b.charity === charityId && !b.isDeleted);

  if (req.query.search) {
    const s = req.query.search.toLowerCase();
    blogs = blogs.filter(
      (b) =>
        b.title.toLowerCase().includes(s) ||
        b.description.toLowerCase().includes(s) ||
        (b.hashtags || []).some((tag) => tag.toLowerCase().includes(s))
    );
  }

  const sortBy = req.query.sortBy || "createdAt";
  const order = req.query.order === "asc" ? 1 : -1;
  const sortFn = (a, b) => {
    const valA = sortBy === "title" ? a[sortBy].toLowerCase() : a[sortBy];
    const valB = sortBy === "title" ? b[sortBy].toLowerCase() : b[sortBy];
    return order * (valA > valB ? 1 : valA < valB ? -1 : 0);
  };
  blogs.sort(sortFn);

  const page = Number(req.query.page) || 1;
  const limit = Math.min(Number(req.query.limit) || 10, 100);
  const { items, pagination } = paginate(blogs, page, limit);

  res.json({
    success: true,
    message: "Blogs fetched successfully",
    blogs: items,
    pagination: { ...pagination, sortBy, order: order === 1 ? "asc" : "desc", search: req.query.search || "" },
  });
});

// 5. Get Charity Ranking (authenticated)
/**
 * Get ranking and impact statistics for the authenticated charity
 * @route   GET /api/v1/charity_organization/charity-ranking
 * @access  Private (charity organization)
 * @header  {string} x-auth-token
 * @returns {200}
 *          {
 *            "success": true,
 *            "message": "Charity ranking retrieved successfully",
 *            "data": {
 *              "ranking": {
 *                "_id": string,
 *                "charityId": string,
 *                "userId": string,
 *                "country": string,
 *                "donationAmount": number,
 *                "impactLife": number,
 *                "donors": [{ "donorId": string, "totalAmount": number, "impactLife": number }, ...],
 *                "createdAt": string,
 *                "updatedAt": string
 *              },
 *              "rank": number
 *            }
 *          }
 * @returns {401}                         Unauthorized
 * @returns {404}                         No ranking data found
 */
app.get("/api/v1/charity_organization/charity-ranking", (req, res) => {
  const token = req.headers["x-auth-token"];
  const charityId = getCharityIdFromToken(token);
  if (!charityId) return res.status(401).json({ success: false, message: "Unauthorized" });

  const ranking = charityrankings.find((r) => r.charityId === charityId);
  if (!ranking) {
    return res.status(404).json({ success: false, message: "Ranking data not found for this charity" });
  }

  const allRanked = [...charityrankings].sort((a, b) => b.donationAmount - a.donationAmount);
  const rank = allRanked.findIndex((r) => r.charityId === charityId) + 1;

  res.json({
    success: true,
    message: "Charity ranking retrieved successfully",
    data: { ranking, rank },
  });
});

// ───────────────────────────────────────────────
// Start Server
// ───────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`✅ Node API running on http://localhost:${PORT}`);
  console.log(`   Tool endpoint:   http://localhost:${PORT}/api/stats?q=charity_donor_count`);
  console.log(`   Search:          http://localhost:${PORT}/api/v1/charity_organization/search?search=helping`);
  console.log(`   Profile:         http://localhost:${PORT}/api/v1/charity_organization/get-charity-profile/${ORG_1}`);
  console.log(`   Products (auth): http://localhost:${PORT}/api/v1/products/get-charity-products`);
  console.log(`                    Header → x-auth-token: ${AUTH_TOKEN}`);
});