const express = require("express");

const app = express();
const PORT = 3000;

// GET endpoint
// Example call:
// http://localhost:3000/api/stats?q=anything
app.get("/api/stats", (req, res) => {
    const inputString = req.query.q || "";

    // Black-box logic (you said no implementation needed)
    const responseText =
        "We have 20 charities working with us, and we have 1000 volunteers. We have raised $500,000 for our causes.";

    res.json({
        input_received: inputString,
        message: responseText
    });
});

app.listen(PORT, () => {
    console.log(`Node API running on http://localhost:${PORT}`);
});
