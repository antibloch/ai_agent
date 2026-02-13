// server.js

const AGENT_URL = "http://localhost:8000/invoke";

// -------- CLI Handling --------
// Usage:
// node server.js
// node server.js --trace
// node server.js "your custom query"
// node server.js "your query" --trace

const args = process.argv.slice(2);

let includeTrace = false;
let query = null;

for (const arg of args) {
  if (arg === "--trace") {
    includeTrace = true;
  } else {
    query = arg;
  }
}

// Default query if none provided
if (!query) {
  query =
    "Compute pi upto 13 decimal places using python, then search the web for latest Elon Musk net worth and summarize in 1 line.";
}

async function main() {
  const payload = {
    query,
    include_trace: includeTrace,
  };

  try {
    const resp = await fetch(AGENT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      console.error(`Agent server error (${resp.status}):\n${errText}`);
      process.exit(1);
    }

    const data = await resp.json();

    console.log("\n=== FINAL ANSWER ===\n");
    console.log(data.answer);

    if (includeTrace && data.trace) {
      console.log("\n=== TRACE ===\n");

      data.trace.forEach((msg, index) => {
        console.log(`--- Message ${index + 1} ---`);
        console.log(`Type: ${msg.type}`);
        console.log(msg.content);
        console.log();
      });
    }

  } catch (e) {
    console.error("Failed to call agent server:", e?.message ?? e);
    process.exit(1);
  }
}

main();
