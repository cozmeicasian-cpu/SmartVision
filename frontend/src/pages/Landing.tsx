import React from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

export default function Landing() {
    const navigate = useNavigate();

    return (
        <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.6 }}
            className="min-h-screen bg-gray-900 text-white flex flex-col justify-center items-center text-center p-6"
        >
            <h1 className="text-5xl font-extrabold text-indigo-400 mb-4">
                🧠 SmartVision
            </h1>
            <p className="text-gray-300 max-w-xl mb-8 text-lg">
                AI-powered image classifier with Grad-CAM visualization — understand
                what your model sees in real time.
            </p>

            <button
                onClick={() => navigate("/dashboard")}
                className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-8 py-3 rounded-lg shadow-lg transition-all duration-200"
            >
                Go to Dashboard →
            </button>

            <footer className="absolute bottom-6 text-gray-500 text-sm">
                Built with ❤️ using React, FastAPI & PyTorch
            </footer>
        </motion.div>
    );
}
