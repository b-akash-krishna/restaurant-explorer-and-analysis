import { useState } from 'react';
import { TrendingUp, Loader2 } from 'lucide-react';

interface PredictionResult {
  predicted_rating: number;
  features_used: {
    votes: number;
    average_cost: number;
    price_range: number;
    has_table_booking: boolean;
    has_online_delivery: boolean;
  };
}

function RatingPrediction() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    votes: 100,
    average_cost: 500,
    price_range: 2,
    has_table_booking: false,
    has_online_delivery: true,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/predict-rating', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error('Prediction failed');

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Failed to predict rating. Make sure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <TrendingUp className="h-8 w-8 text-blue-600" />
        <h1 className="text-3xl font-bold text-gray-900">Restaurant Rating Prediction</h1>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <p className="text-gray-600 mb-6">
          Predict restaurant ratings using machine learning. Enter restaurant features below to get an estimated rating.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Number of Votes
              </label>
              <input
                type="number"
                value={formData.votes}
                onChange={(e) => setFormData({ ...formData, votes: parseInt(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="0"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Average Cost for Two
              </label>
              <input
                type="number"
                value={formData.average_cost}
                onChange={(e) => setFormData({ ...formData, average_cost: parseFloat(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="0"
                step="0.01"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Price Range (1-4)
              </label>
              <select
                value={formData.price_range}
                onChange={(e) => setFormData({ ...formData, price_range: parseInt(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="1">1 - Budget</option>
                <option value="2">2 - Moderate</option>
                <option value="3">3 - Expensive</option>
                <option value="4">4 - Very Expensive</option>
              </select>
            </div>

            <div className="space-y-3">
              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={formData.has_table_booking}
                  onChange={(e) => setFormData({ ...formData, has_table_booking: e.target.checked })}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <span className="text-sm font-medium text-gray-700">Has Table Booking</span>
              </label>

              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={formData.has_online_delivery}
                  onChange={(e) => setFormData({ ...formData, has_online_delivery: e.target.checked })}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <span className="text-sm font-medium text-gray-700">Has Online Delivery</span>
              </label>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-3 rounded-md font-medium hover:bg-blue-700 transition-colors disabled:bg-blue-300 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin h-5 w-5 mr-2" />
                Predicting...
              </>
            ) : (
              'Predict Rating'
            )}
          </button>
        </form>

        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        )}

        {result && (
          <div className="mt-6 p-6 bg-blue-50 border border-blue-200 rounded-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Result</h3>
            <div className="flex items-center justify-center mb-4">
              <div className="text-center">
                <div className="text-5xl font-bold text-blue-600">{result.predicted_rating}</div>
                <div className="text-sm text-gray-600 mt-2">Predicted Rating (out of 5)</div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-blue-200">
              <p className="text-sm text-gray-600">
                Based on {result.features_used.votes} votes, average cost of {result.features_used.average_cost},
                price range {result.features_used.price_range},
                {result.features_used.has_table_booking ? ' with' : ' without'} table booking, and
                {result.features_used.has_online_delivery ? ' with' : ' without'} online delivery.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default RatingPrediction;
