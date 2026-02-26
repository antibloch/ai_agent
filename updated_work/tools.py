import json
import os
from typing import Any, Dict, Optional

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_experimental.tools import PythonREPLTool
from langchain_mcp_adapters.client import MultiServerMCPClient


# --------------------------
# Common helpers
# --------------------------

DEFAULT_BASE_URL = "http://localhost:3000"
DEFAULT_AUTH_TOKEN = "charity-demo-token-2026"


def _ok(result: Any, **meta) -> str:
    payload = {"ok": True, "result": result}
    if meta:
        payload["meta"] = meta
    return json.dumps(payload, ensure_ascii=False, default=str)


def _fail(error: str, **meta) -> str:
    payload = {"ok": False, "error": error}
    if meta:
        payload["meta"] = meta
    return json.dumps(payload, ensure_ascii=False, default=str)


def _get(url: str, *, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 10) -> Dict[str, Any]:
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
    # Your Node /api/stats always returns 200 even on errors,
    # but other endpoints return 4xx; handle both.
    try:
        data = r.json()
    except Exception:
        data = {"raw_text": r.text}
    return {"status": r.status_code, "json": data}


# --------------------------
# 1) Legacy stats tool (/api/stats?q=...)
# --------------------------

def build_node_stats_tool(base_url: str = DEFAULT_BASE_URL) -> StructuredTool:
    CANONICAL_TOOLS = [
        "charity_donor_count", "charity_impactlife", "charity_donor_amount",
        "charity_total_donation", "charity_items_category",
        "charity_product_price_description", "charity_blogs",
        "charity_address", "charity_country_availability", "charity_contact_info",
    ]

    def call_node_stats(tool_name: str) -> str:
        tool_name = (tool_name or "").strip()
        if not tool_name:
            return _fail("Tool name is required.", valid_tools=CANONICAL_TOOLS)
        if tool_name not in CANONICAL_TOOLS:
            return _fail("Invalid tool name.", provided=tool_name, valid_tools=CANONICAL_TOOLS)

        try:
            out = _get(f"{base_url}/api/stats", params={"q": tool_name})
            # /api/stats returns {ok: true/false, tool, query, data, ...}
            return _ok(out["json"], endpoint="/api/stats", http_status=out["status"])
        except requests.RequestException as e:
            return _fail(str(e), tool=tool_name, endpoint="/api/stats")

    class CharityStatsInput(BaseModel):
        tool_name: str = Field(..., description="Exact tool name from the allowed list in the tool description")

    return StructuredTool.from_function(
        func=call_node_stats,
        name="get_charity_stats",
        description=(
            "Fetch internal charity data from Node-js server via /api/stats.\n"
            "Argument tool_name must be one of:\n" + "\n".join([f"- {t}" for t in CANONICAL_TOOLS])
        ),
        args_schema=CharityStatsInput,
    )


# --------------------------
# 2) Search charities (public)
# GET /api/v1/charity_organization/search?search=...
# --------------------------

def build_search_charities_tool(base_url: str = DEFAULT_BASE_URL) -> StructuredTool:
    def search_charities(search: str) -> str:
        s = (search or "").strip()
        if not s:
            return _fail("search is required.")
        try:
            out = _get(f"{base_url}/api/v1/charity_organization/search", params={"search": s})
            if out["status"] >= 400:
                return _fail(f"HTTP {out['status']}: {out['json']}", endpoint="/api/v1/charity_organization/search")
            return _ok(out["json"], endpoint="/api/v1/charity_organization/search", http_status=out["status"])
        except requests.RequestException as e:
            return _fail(str(e), endpoint="/api/v1/charity_organization/search")

    class SearchInput(BaseModel):
        search: str = Field(..., description="Search term (charity name or email).")

    return StructuredTool.from_function(
        func=search_charities,
        name="search_charities",
        description=(
                "Search approved charities by name or email (PUBLIC).\n"
                "Endpoint: GET /api/v1/charity_organization/search?search=<term>\n"
                "Auth: None\n\n"
                "Args (JSON):\n"
                '- {"search": "<string>"}  (required)\n\n'
                "Response envelope (tool wrapper):\n"
                "- ok=true  -> { ok: true, result: <server_json>, meta?: {...} }\n"
                "- ok=false -> { ok: false, error: <string>, meta?: {...} }\n\n"
                "Server JSON (success):\n"
                "{ success: true, message: string, data: { searchQuery: string, totalResults: number, charities: ["
                "{ _id: string, name: string, email: string, logo: string|null, address: object, verificationStatus: string }"
                "] } }\n"
                "Server JSON (missing search): returns HTTP 400 { success:false, message:'Search query is required' }."
                ),
        args_schema=SearchInput,
    )


# --------------------------
# 3) Get charity profile by ID (public)
# GET /api/v1/charity_organization/get-charity-profile/:charityId
# --------------------------

def build_get_charity_profile_tool(base_url: str = DEFAULT_BASE_URL) -> StructuredTool:
    def get_charity_profile(charity_id: str) -> str:
        cid = (charity_id or "").strip()
        if not cid:
            return _fail("charity_id is required.")
        try:
            out = _get(f"{base_url}/api/v1/charity_organization/get-charity-profile/{cid}")
            if out["status"] >= 400:
                return _fail(f"HTTP {out['status']}: {out['json']}", endpoint="/api/v1/charity_organization/get-charity-profile/:id")
            return _ok(out["json"], endpoint="/api/v1/charity_organization/get-charity-profile/:id", http_status=out["status"])
        except requests.RequestException as e:
            return _fail(str(e), endpoint="/api/v1/charity_organization/get-charity-profile/:id")

    class ProfileInput(BaseModel):
        charity_id: str = Field(..., description="Charity organization ID (e.g., org_001).")

    return StructuredTool.from_function(
        func=get_charity_profile,
        name="get_charity_profile",
        description=(
            "Get detailed profile for a single charity by charity_id (PUBLIC).\n"
            "Endpoint: GET /api/v1/charity_organization/get-charity-profile/:charityId\n"
            "Auth: None\n\n"
            "Args (JSON):\n"
            '- {"charity_id": "<string>"}  (required, e.g., "org_001")\n\n'
            "Response envelope (tool wrapper):\n"
            "- ok=true  -> { ok: true, result: <server_json>, meta?: {...} }\n"
            "- ok=false -> { ok: false, error: <string>, meta?: {...} }\n\n"
            "Server JSON (success):\n"
            "{ success:true, message:string, charity:{ _id,name,email,phone,logo,address,registrationNumber,"
            "verificationStatus,description,website,createdAt,updatedAt } }\n"
            "Server JSON (not found): HTTP 404 { success:false, message:'Charity not found or not approved' }."
            ),
        args_schema=ProfileInput,
    )


# --------------------------
# 4) Get charity products (auth)
# GET /api/v1/products/get-charity-products
# headers: x-auth-token
# --------------------------

def build_get_charity_products_tool(
    base_url: str = DEFAULT_BASE_URL,
    default_token: str = DEFAULT_AUTH_TOKEN,
) -> StructuredTool:
    def get_charity_products(
        page: int = 1,
        limit: int = 10,
        isActive: Optional[bool] = None,
        isDeleted: Optional[bool] = None,
        status: Optional[str] = None,
        productId: Optional[str] = None,
        minPrice: Optional[float] = None,
        maxPrice: Optional[float] = None,
        startDate: Optional[str] = None,
        endDate: Optional[str] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
        sort: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> str:
        token = default_token
        if not token:
            return _fail("auth_token is required for this endpoint (x-auth-token header).")

        params: Dict[str, Any] = {"page": page, "limit": limit}
        if isActive is not None:
            params["isActive"] = str(bool(isActive)).lower()
        if isDeleted is not None:
            params["isDeleted"] = str(bool(isDeleted)).lower()
        if status:
            params["status"] = status
        if productId:
            params["productId"] = productId
        if minPrice is not None:
            params["minPrice"] = minPrice
        if maxPrice is not None:
            params["maxPrice"] = maxPrice
        if startDate:
            params["startDate"] = startDate
        if endDate:
            params["endDate"] = endDate
        if category:
            params["category"] = category
        if search:
            params["search"] = search
        if sort:
            params["sort"] = sort

        try:
            out = _get(
                f"{base_url}/api/v1/products/get-charity-products",
                params=params,
                headers={"x-auth-token": token},
            )
            if out["status"] >= 400:
                return _fail(f"HTTP {out['status']}: {out['json']}", endpoint="/api/v1/products/get-charity-products")
            return _ok(out["json"], endpoint="/api/v1/products/get-charity-products", http_status=out["status"])
        except requests.RequestException as e:
            return _fail(str(e), endpoint="/api/v1/products/get-charity-products")

    class ProductsInput(BaseModel):
        page: int = Field(1, ge=1)
        limit: int = Field(10, ge=1, le=100)
        isActive: Optional[bool] = None
        isDeleted: Optional[bool] = None
        status: Optional[str] = Field(None, description="approved|pending|rejected")
        productId: Optional[str] = None
        minPrice: Optional[float] = None
        maxPrice: Optional[float] = None
        startDate: Optional[str] = Field(None, description="ISO date string")
        endDate: Optional[str] = Field(None, description="ISO date string")
        category: Optional[str] = Field(None, description="comma-separated category IDs")
        search: Optional[str] = None
        sort: Optional[str] = Field(None, description="e.g. -createdAt, price, name")
        auth_token: Optional[str] = Field(None, description="x-auth-token header; defaults to CHARITY_AUTH_TOKEN env var if set")

    return StructuredTool.from_function(
        func=get_charity_products,
        name="get_charity_products",
        description=(
            "Get paginated products belonging to the AUTHENTICATED charity.\n"
            "Endpoint: GET /api/v1/products/get-charity-products\n"
            "Auth: Required header x-auth-token.\n\n"
            "CRITICAL: Always include page and limit in every call "
            "(defaults: page=1, limit=10). This prevents huge responses and is required "
            "for reliable pagination.\n\n"
            "Args (JSON):\n"
            '- {"auth_token":"<string>", "page":1, "limit":10, "isActive":true/false, "isDeleted":true/false,\n'
            '   "status":"approved|pending|rejected", "productId":"<string>", "minPrice":number, "maxPrice":number,\n'
            '   "startDate":"<ISO date>", "endDate":"<ISO date>", "category":"cat_1,cat_2", "search":"<string>",\n'
            '   "sort":"-createdAt|price|name"}\n'
            "auth_token is required; page & limit should always be present; others optional.\n\n"
            "Response envelope (tool wrapper):\n"
            "- ok=true  -> { ok: true, result: <server_json>, meta?: {...} }\n"
            "- ok=false -> { ok: false, error: <string>, meta?: {...} }\n\n"
            "Server JSON (success):\n"
            "{ success:true, message:string, data:{ products:[{ _id,name,description,pricePerUnit,isActive,status,"
            "charity:{_id,name,logo}, parent|null, createdAt, updatedAt }], pagination:{ total,page,limit,totalPages,"
            "hasNext,hasPrev } } }\n"
            "Server JSON (unauthorized): HTTP 401 { success:false, message:'Unauthorized' }."
            ),
        args_schema=ProductsInput,
    )


# --------------------------
# 5) Get charity blogs (auth)
# GET /api/v1/charity_organization/blogs
# --------------------------

def build_get_charity_blogs_tool(
    base_url: str = DEFAULT_BASE_URL,
    default_token: str = DEFAULT_AUTH_TOKEN,
) -> StructuredTool:
    def get_charity_blogs(
        page: int = 1,
        limit: int = 10,
        search: Optional[str] = None,
        sortBy: Optional[str] = None,
        order: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> str:
        token = default_token
        if not token:
            return _fail("auth_token is required for this endpoint (x-auth-token header).")

        params: Dict[str, Any] = {"page": page, "limit": limit}
        if search:
            params["search"] = search
        if sortBy:
            params["sortBy"] = sortBy
        if order:
            params["order"] = order

        try:
            out = _get(
                f"{base_url}/api/v1/charity_organization/blogs",
                params=params,
                headers={"x-auth-token": token},
            )
            if out["status"] >= 400:
                return _fail(f"HTTP {out['status']}: {out['json']}", endpoint="/api/v1/charity_organization/blogs")
            return _ok(out["json"], endpoint="/api/v1/charity_organization/blogs", http_status=out["status"])
        except requests.RequestException as e:
            return _fail(str(e), endpoint="/api/v1/charity_organization/blogs")

    class BlogsInput(BaseModel):
        page: int = Field(1, ge=1)
        limit: int = Field(10, ge=1, le=100)
        search: Optional[str] = Field(None, description="Search title/description/hashtags")
        sortBy: Optional[str] = Field(None, description="createdAt|updatedAt|title|status")
        order: Optional[str] = Field(None, description="asc|desc")
        auth_token: Optional[str] = Field(None, description="x-auth-token header; defaults to CHARITY_AUTH_TOKEN env var if set")

    return StructuredTool.from_function(
        func=get_charity_blogs,
        name="get_charity_blogs",
        description=(
            "Get paginated blogs belonging to the AUTHENTICATED charity.\n"
            "Endpoint: GET /api/v1/charity_organization/blogs\n"
            "Auth: Required header x-auth-token.\n\n"
            "CRITICAL: Always include page and limit in every call "
            "(defaults: page=1, limit=10). This is required for reliable pagination.\n\n"
            "Args (JSON):\n"
            '- {"auth_token":"<string>", "page":1, "limit":10, "search":"<string>", '
            '"sortBy":"createdAt|updatedAt|title|status", "order":"asc|desc"}\n'
            "auth_token is required; page & limit should always be present; others optional.\n\n"
            "Response envelope (tool wrapper):\n"
            "- ok=true  -> { ok: true, result: <server_json>, meta?: {...} }\n"
            "- ok=false -> { ok: false, error: <string>, meta?: {...} }\n\n"
            "Server JSON (success):\n"
            "{ success:true, message:string, blogs:[{ _id,charity,title,description,hashtags,file,status,isDeleted,createdAt,updatedAt }],\n"
            "  pagination:{ total,page,limit,totalPages,hasNext,hasPrev,sortBy,order,search } }\n"
            "Server JSON (unauthorized): HTTP 401 { success:false, message:'Unauthorized' }."
            ),
        args_schema=BlogsInput,
    )


# --------------------------
# 6) Get charity ranking (auth)
# GET /api/v1/charity_organization/charity-ranking
# --------------------------

def build_get_charity_ranking_tool(
    base_url: str = DEFAULT_BASE_URL,
    default_token: str = DEFAULT_AUTH_TOKEN,
) -> StructuredTool:
    def get_charity_ranking(auth_token: Optional[str] = None) -> str:
        token = default_token
        if not token:
            return _fail("auth_token is required for this endpoint (x-auth-token header).")
        try:
            out = _get(
                f"{base_url}/api/v1/charity_organization/charity-ranking",
                headers={"x-auth-token": token},
            )
            if out["status"] >= 400:
                return _fail(f"HTTP {out['status']}: {out['json']}", endpoint="/api/v1/charity_organization/charity-ranking")
            return _ok(out["json"], endpoint="/api/v1/charity_organization/charity-ranking", http_status=out["status"])
        except requests.RequestException as e:
            return _fail(str(e), endpoint="/api/v1/charity_organization/charity-ranking")

    class RankingInput(BaseModel):
        auth_token: Optional[str] = Field(None, description="x-auth-token header; defaults to CHARITY_AUTH_TOKEN env var if set")

    return StructuredTool.from_function(
        func=get_charity_ranking,
        name="get_charity_ranking",
        description=(
                "Get ranking + impact stats for the AUTHENTICATED charity.\n"
                "Endpoint: GET /api/v1/charity_organization/charity-ranking\n"
                "Auth: Required header x-auth-token.\n\n"
                "Args (JSON):\n"
                '- {"auth_token":"<string>"} (required)\n\n'
                "Response envelope (tool wrapper):\n"
                "- ok=true  -> { ok: true, result: <server_json>, meta?: {...} }\n"
                "- ok=false -> { ok: false, error: <string>, meta?: {...} }\n\n"
                "Server JSON (success):\n"
                "{ success:true, message:string, data:{ ranking:{ _id,charityId,userId,country,donationAmount,impactLife,donors:[...],createdAt,updatedAt }, rank:number } }\n"
                "Server JSON (unauthorized): HTTP 401 { success:false, message:'Unauthorized' }.\n"
                "Server JSON (no ranking): HTTP 404 { success:false, message:'Ranking data not found for this charity' }."
                ),
        args_schema=RankingInput,
    )


# --------------------------
# Tool setup
# --------------------------

async def setup_tools():
    local_tools = [
        build_node_stats_tool(),
        build_search_charities_tool(),
        build_get_charity_profile_tool(),
        build_get_charity_products_tool(),
        build_get_charity_blogs_tool(),
        build_get_charity_ranking_tool(),
        PythonREPLTool(),
    ]

    client = MultiServerMCPClient({
        "fetch": {"transport": "stdio", "command": "npx", "args": ["-y", "fetcher-mcp"]}
    })
    mcp_tools = await client.get_tools()
    return [*local_tools, *mcp_tools]




def build_tool_context(tools_by_name: dict):
    blocks = []
    for tool in tools_by_name.values():
        name = tool.name
        description = (getattr(tool, "description", "") or "No description.").strip()
        args_schema = getattr(tool, "args_schema", None)
        if args_schema and hasattr(args_schema, "model_fields"):
            fields = args_schema.model_fields
            arg_lines = []
            for k, v in fields.items():
                req = getattr(v, "is_required", lambda: False)()
                arg_lines.append(f"- {k} ({'required' if req else 'optional'})")
            args_text = "\n".join(arg_lines) if arg_lines else "No parameters"
        else:
            args_text = "- input (required string). For Python_REPL, this must be python code."
        blocks.append(f"""
{name}
Description:
{description}

Arguments:
{args_text}
""")
    return "\n\n".join(blocks)