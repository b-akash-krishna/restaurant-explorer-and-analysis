import { useState } from 'react';
import { Utensils, Loader2 } from 'lucide-react';

interface ClassificationResult {
  predicted_cuisine: string;
  confidence: number;
  features_used: {
    aggregate_rating: number;
    votes: number;
    price_range: number;
    average_cost: number;
    has_table_booking: boolean;
    has_online_delivery: boolean;
  };
}

function CuisineClassification() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    aggregate_rating: 4.0,
    votes: 150,
    price_range: 2,
    average_cost: 600,
    has_table_booking: true,
    has_online_delivery: false,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/classify-cuisine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error('Classification failed');

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Failed to classify cuisine. Make sure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <Utensils className="h-8 w-8 text-orange-600" />
        <h1 className="text-3xl font-bold text-gray-900">Cuisine Classification</h1>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <p className="text-gray-600 mb-6">
          Classify restaurant cuisines using multi-class classification. Enter restaurant characteristics to predict the cuisine type.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Aggregate Rating (0-5)
              </label>
              <input
                type="number"
                value={formData.aggregate_rating}
                onChange={(e) => setFormData({ ...formData, aggregate_rating: parseFloat(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                min="0"
                max="5"
                step="0.1"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Number of Votes
              </label>
              <input
                type="number"
                value={formData.votes}
                onChange={(e) => setFormData({ ...formData, votes: parseInt(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                min="0"
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
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                required
              >
                <option value="1">1 - Budget</option>
                <option value="2">2 - Moderate</option>
                <option value="3">3 - Expensive</option>
                <option value="4">4 - Very Expensive</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Average Cost for Two
              </label>
              <input
                type="number"
                value={formData.average_cost}
                onChange={(e) => setFormData({ ...formData, average_cost: parseFloat(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                min="0"
                step="0.01"
                required
              />
            </div>

            <div className="space-y-3 md:col-span-2">
              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={formData.has_table_booking}
                  onChange={(e) => setFormData({ ...formData, has_table_booking: e.target.checked })}
                  className="w-4 h-4 text-orange-600 border-gray-300 rounded focus:ring-orange-500"
                />
                <span className="text-sm font-medium text-gray-700">Has Table Booking</span>
              </label>

              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={formData.has_online_delivery}
                  onChange={(e) => setFormData({ ...formData, has_online_delivery: e.target.checked })}
                  className="w-4 h-4 text-orange-600 border-gray-300 rounded focus:ring-orange-500"
                />
                <span className="text-sm font-medium text-gray-700">Has Online Delivery</span>
              </label>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-orange-600 text-white py-3 rounded-md font-medium hover:bg-orange-700 transition-colors disabled:bg-orange-300 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin h-5 w-5 mr-2" />
                Classifying...
              </>
            ) : (
              'Classify Cuisine'
            )}
          </button>
        </form>

        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        )}

        {result && (
          <div className="mt-6 p-6 bg-orange-50 border border-orange-200 rounded-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Classification Result</h3>
            <div className="text-center mb-4">
              <div className="text-4xl font-bold text-orange-600 mb-2">
                {result.predicted_cuisine}
              </div>
              <div className="text-sm text-gray-600">Predicted Cuisine Type</div>
            </div>

            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">Confidence Level</span>
                <span className="text-sm font-semibold text-orange-600">{result.confidence}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-orange-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${result.confidence}%` }}
                ></div>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-orange-200">
              <p className="text-sm text-gray-600 mb-2">Classification based on:</p>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>Rating: {result.features_used.aggregate_rating} with {result.features_used.votes} votes</li>
                <li>Average Cost: {result.features_used.average_cost} (Price Range: {result.features_used.price_range})</li>
                <li>Table Booking: {result.features_used.has_table_booking ? 'Available' : 'Not Available'}</li>
                <li>Online Delivery: {result.features_used.has_online_delivery ? 'Available' : 'Not Available'}</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default CuisineClassification;
