// src/components/SpamClassifier.js

import React, { useState } from "react";
import axios from "axios";
import { FaRedoAlt } from "react-icons/fa"; // Import the reset icon from react-icons

const SpamClassifier = () => {
  const [emailSubject, setEmailSubject] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState("");

  const handleCheckSpam = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/predict/", {
        email_subject: emailSubject,
      });
      setTimeout(() => {
        setPrediction(response.data.prediction);
        setIsLoading(false);
      }, 2000);
    } catch (error) {
      console.error("Error checking spam:", error);
      setPrediction("Error occurred");
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setEmailSubject("");
    setPrediction("");
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 p-8">
      <div className="w-full max-w-md p-8 bg-gray-800 rounded-lg shadow-xl border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white">Spam Classifier</h2>
          <button
            onClick={handleReset}
            className="text-gray-400 hover:text-white transition duration-300 ease-in-out"
          >
            <FaRedoAlt size={20} />
          </button>
        </div>
        <input
          type="text"
          placeholder="Enter email subject"
          value={emailSubject}
          onChange={(e) => setEmailSubject(e.target.value)}
          className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg mb-6 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 transition duration-300 ease-in-out"
          list="email-subjects"
        />
        <datalist id="email-subjects">
          <option value="Win money now!" />
          <option value="Cheap loans available with low interest rates!" />
          <option value="Get rich quick with this exclusive offer!" />
          <option value="Earn money while you sleep with this easy method!" />
          <option value="Important information about your account update." />
          <option value="Meeting on Friday to discuss the new project." />
          <option value="Your invoice is ready for download." />
          <option value="Team lunch tomorrow at 12 PM!" />
        </datalist>
        <button
          disabled={isLoading || emailSubject.length < 1}
          onClick={handleCheckSpam}
          className="w-full px-4 py-3 flex items-center justify-center gap-3 bg-blue-500 text-white font-semibold rounded-lg hover:bg-blue-600 disabled:bg-gray-700 transition duration-300 ease-in-out"
        >
          {isLoading ? (
            <div className="flex items-center">
              <div className="w-5 h-5 border-4 border-blue-500 border-t-transparent border-solid rounded-full animate-spin"></div>
            </div>
          ) : (
            "Check"
          )}
        </button>
        {prediction && (
          <div className="mt-6 text-center">
          {prediction === "spam" && (
            <div className="p-4 rounded-lg bg-red-900 text-red-400">
              <p className="text-lg font-medium">🚨 This email is spam!</p>
            </div>
          )}
          {prediction === "not spam" && (
            <div className="p-4 rounded-lg bg-green-900 text-green-400">
              <p className="text-lg font-medium">✅ This email is not spam.</p>
            </div>
          )}
          {prediction === "Error occurred" && (
            <div className="p-4 rounded-lg bg-gray-800 text-gray-400">
              <p className="text-lg font-medium">⚠️ An error occurred. Please try again.</p>
            </div>
          )}
        </div>
        )}
      </div>
    </div>
  );
};

export default SpamClassifier;
