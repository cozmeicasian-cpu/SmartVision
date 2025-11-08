import React, { useState } from "react";
import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";

export default function Dashboard() {
    const [file, setFile] = useState<File | null>(null);
    const [prediction, setPrediction] = useState<string>("");
    const [heatmap, setHeatmap] = useState<string>("");
    const [probabilities, setProbabilities] = useState<{ name: string; value: number }[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>("");
    const [confidence, setConfidence] = useState<number | null>(null);


    // ✅ Use environment variable with fallback
    const API_BASE = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

    const handleUpload = async () => {
        if (!file) {
            alert("Please upload an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            setLoading(true);
            setError("");

            const res = await axios.post(`${API_BASE}/predict`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            setPrediction(res.data.prediction);
            setHeatmap(`data:image/png;base64,${res.data.heatmap}`);
            if (res.data.confidence) {
                setConfidence(res.data.confidence);
            }


            if (res.data.probabilities) {
                const chartData = res.data.probabilities.map((p: [string, number]) => ({
                    name: p[0],
                    value: Math.round(p[1] * 100),
                }));
                setProbabilities(chartData);
            } else {
                setProbabilities([]);
            }
        } catch (err: any) {
            console.error("Prediction failed:", err);
            setError("Prediction failed. Please check if the backend is running or reachable.");
        } finally {
            setLoading(false);
        }
    };

    const handleDownload = () => {
        if (!heatmap) return;
        const link = document.createElement("a");
        link.href = heatmap;
        link.download = `${prediction || "heatmap"}_gradcam.png`;
        link.click();
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.6 }}
            className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-6"
        >
            <header className="text-center mb-10">
                <h1 className="text-4xl font-bold text-indigo-400">🧠 SmartVision Dashboard</h1>
                <p className="text-gray-400 mt-2">
                    Upload an image to generate a Grad-CAM visualization.
                </p>
            </header>

            <div className="bg-gray-800 rounded-2xl shadow-xl p-8 w-full max-w-4xl">
                <div className="flex flex-col items-center space-y-4">
                    <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => setFile(e.target.files?.[0] || null)}
                        className="block w-full text-sm text-gray-300 border border-gray-700 rounded-lg cursor-pointer bg-gray-900 focus:outline-none"
                    />
                    <button
                        onClick={handleUpload}
                        disabled={loading}
                        className={`px-6 py-2 rounded-lg font-semibold text-white ${loading ? "bg-gray-600" : "bg-indigo-500 hover:bg-indigo-600"
                            }`}
                    >
                        {loading ? "Predicting..." : "Predict"}
                    </button>
                </div>

                {error && (
                    <p className="text-red-400 text-center mt-4">{error}</p>
                )}

                {prediction && (
                    <div className="mt-8 grid md:grid-cols-2 gap-8">
                        <div className="flex flex-col items-center space-y-4">
                            <h2 className="text-2xl font-bold text-indigo-400">Prediction</h2>
                            <p className="text-4xl font-semibold">
                                {prediction}
                                {confidence !== null && (
                                    <span className="text-lg text-gray-400 ml-2">
                                        ({(confidence * 100).toFixed(1)}%)
                                    </span>
                                )}
                            </p>



                            {probabilities.length > 0 && (
                                <ResponsiveContainer width="100%" height={200}>
                                    <BarChart data={probabilities}>
                                        <XAxis dataKey="name" stroke="#ccc" />
                                        <YAxis stroke="#ccc" />
                                        <Tooltip />
                                        <Bar dataKey="value" fill="#6366F1" />
                                    </BarChart>
                                </ResponsiveContainer>
                            )}
                        </div>

                        <div className="flex flex-col items-center space-y-4">
                            <h2 className="text-2xl font-bold text-indigo-400">Grad-CAM Heatmap</h2>
                            {heatmap ? (
                                <>
                                    <img
                                        src={heatmap}
                                        alt="Grad-CAM heatmap"
                                        className="rounded-lg shadow-md"
                                    />
                                    <button
                                        onClick={handleDownload}
                                        className="mt-2 text-sm bg-indigo-600 hover:bg-indigo-700 px-4 py-1 rounded-lg"
                                    >
                                        Download Heatmap
                                    </button>
                                </>
                            ) : (
                                <p className="text-gray-400">No image yet</p>
                            )}
                        </div>
                    </div>
                )}
            </div>
            
            <footer className="mt-10 text-gray-500 text-sm">
                <button
                    onClick={() => (window.location.href = "/")}
                    className="text-indigo-400 hover:underline"
                >
                    ← Back to Home
                </button>
            </footer>
        </motion.div>
    );
}
