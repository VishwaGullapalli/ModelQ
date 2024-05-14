const express = require("express");
const cors = require("cors");
require("dotenv").config();
const app = express();
const port = process.env.PORT;
const { exec } = require("child_process");
app.use(express.json());
app.use(cors());
app.listen(port, "0.0.0.0", () => {
  console.log(`Server is running on port: ${port}`);
});

app.post("/api/shor", async (req, res) => {
  const inputData = req.body.testCase; // Extract test case from the request body
  console.log(inputData);

  if (!inputData) {
    res.status(400).send("Test case is required");
    return;
  }

  // Send input data to the Python script and receive the predictions
  const pythonProcess = exec("python Shor.py", (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing Python script: ${error}`);
      res.status(500).send("Error executing Python script");
      return;
    }

    const [statement, p, q, time_taken] = stdout.trim().split("\n");
    res.json({ statement, p, q, time_taken });
  });

  pythonProcess.stdin.write(inputData + "\n");
  pythonProcess.stdin.end();
});

app.post("/api/grovers", async (req, res) => {
  const inputData = req.body.testCase; // Extract test case from the request body
  console.log(inputData);

  if (!inputData) {
    res.status(400).send("Test case is required");
    return;
  }

  // Send input data to the Python script and receive the predictions
  const pythonProcess = exec("python Grovers.py", (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing Python script: ${error}`);
      res.status(500).send("Error executing Python script");
      return;
    }

    const [
      MarkedBitstrings,
      // Circuit,
      SampledResults,
      MostCommonBitstring,
      TimeTaken,
    ] = stdout.trim().split("\n");
    res.json({
      MarkedBitstrings,
      // Circuit,
      SampledResults,
      MostCommonBitstring,
      TimeTaken,
    });
  });

  pythonProcess.stdin.write(inputData + "\n");
  pythonProcess.stdin.end();
});

app.post("/api/grovers-framework", async (req, res) => {
  const inputData = req.body.testCase; // Extract test case from the request body
  console.log(inputData);

  if (!inputData) {
    res.status(400).send("Test case is required");
    return;
  }

  // Send input data to the Python script and receive the predictions
  const pythonProcess = exec("python Grovers.py", (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing Python script: ${error}`);
      res.status(500).send("Error executing Python script");
      return;
    }

    const [
      MarkedBitstrings,
      // Circuit,
      SampledResults,
      MostCommonBitstring,
      TimeTaken,
    ] = stdout.trim().split("\n");
    res.json({
      MarkedBitstrings,
      // Circuit,
      SampledResults,
      MostCommonBitstring,
      TimeTaken,
    });
  });

  pythonProcess.stdin.write(inputData + "\n");
  pythonProcess.stdin.end();
});