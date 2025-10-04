import { useState } from 'react';
import { TrendingUp, Loader2 } from 'lucide-react';
import { Skeleton } from "@/components/ui/skeleton";

interface PredictionResult {
  predicted_rating: number;
  features_used: {
    votes: number;
    online_order: number;
    book_table: number;
    location: string;
    rest_type: string;
    cuisines: string;
    cost: number;
  };
}

// A simple skeleton component for the result display
const ResultSkeleton = () => (
  <div className="mt-6 p-6 bg-gray-100 rounded-md">
    <Skeleton className="h-6 w-1/3 mb-4" />
    <div className="flex items-center justify-center mb-4">
      <Skeleton className="h-24 w-24 rounded-full" />
    </div>
    <div className="mt-4 pt-4 border-t border-gray-200">
      <Skeleton className="h-4 w-full mb-2" />
      <Skeleton className="h-4 w-full" />
    </div>
  </div>
);

function RatingPrediction() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    votes: 100,
    online_order: 1,
    book_table: 0,
    location: 'Koramangala 5th Block',
    rest_type: 'Casual Dining',
    cuisines: 'North Indian, Chinese',
    cost: 500,
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? (e.target as HTMLInputElement).checked : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setSuccess(null);

    try {
      // Mapping boolean to integer for the backend model
      const payload = {
        ...formData,
        online_order: formData.online_order ? 1 : 0,
        book_table: formData.book_table ? 1 : 0,
        votes: Number(formData.votes),
        cost: Number(formData.cost),
      };

      const response = await fetch('http://localhost:8000/api/predict-rating', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Assuming you have a way to get the auth token, otherwise this might be 401
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`, 
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Authentication failed. Please log in again.');
        }
        if (response.status === 422) {
          const errorData = await response.json();
          const detailedMessage = errorData.detail 
            ? errorData.detail.map((err: any) => `${err.loc.join(' -> ')}: ${err.msg}`).join('; ')
            : 'Invalid input data. Please check your form and try again.';
          throw new Error(`Validation Error: ${detailedMessage}`);
        }
        throw new Error(`Prediction failed with status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      setSuccess('Prediction successful!');
      // Reset form data after successful prediction
      setFormData({
        votes: 100,
        online_order: 1,
        book_table: 0,
        location: 'Koramangala 5th Block',
        rest_type: 'Casual Dining',
        cuisines: 'North Indian, Chinese',
        cost: 500,
      });

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
      <div className="bg-white shadow-xl rounded-xl p-8 w-full max-w-2xl">
        <div className="flex items-center justify-center mb-6">
          <TrendingUp className="w-12 h-12 text-blue-500" />
        </div>
        <h2 className="text-3xl font-bold text-center text-gray-800 mb-2">Predict Rating</h2>
        <p className="text-center text-gray-500 mb-8">
          Enter restaurant features to predict its aggregate rating.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="votes" className="block text-sm font-medium text-gray-700">
              Votes
            </label>
            <input
              type="number"
              id="votes"
              name="votes"
              value={formData.votes}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            />
          </div>

          <div>
            <label htmlFor="online_order" className="block text-sm font-medium text-gray-700">
              Online Order
            </label>
            <select
              id="online_order"
              name="online_order"
              value={formData.online_order.toString()}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          <div>
            <label htmlFor="book_table" className="block text-sm font-medium text-gray-700">
              Book Table
            </label>
            <select
              id="book_table"
              name="book_table"
              value={formData.book_table.toString()}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          <div>
            <label htmlFor="cost" className="block text-sm font-medium text-gray-700">
              Average Cost for two
            </label>
            <input
              type="number"
              id="cost"
              name="cost"
              value={formData.cost}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            />
          </div>

          <div>
            <label htmlFor="location" className="block text-sm font-medium text-gray-700">
              Location
            </label>
            <input
              type="text"
              id="location"
              name="location"
              value={formData.location}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            />
          </div>

          <div>
            <label htmlFor="rest_type" className="block text-sm font-medium text-gray-700">
              Restaurant Type
            </label>
            <input
              type="text"
              id="rest_type"
              name="rest_type"
              value={formData.rest_type}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            />
          </div>

          <div>
            <label htmlFor="cuisines" className="block text-sm font-medium text-gray-700">
              Cuisines
            </label>
            <input
              type="text"
              id="cuisines"
              name="cuisines"
              value={formData.cuisines}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            />
          </div>

          <button
            type="submit"
            className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
              loading ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
            }`}
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              'Predict Rating'
            )}
          </button>
        </form>

        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-600 text-sm">
              <span className="font-bold">Error:</span> {error}
            </p>
          </div>
        )}

        {success && (
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-md">
            <p className="text-green-600 text-sm">
              <span className="font-bold">Success:</span> {success}
            </p>
          </div>
        )}
        
        {loading && !result && <ResultSkeleton />}

        {!loading && result && (
          <div className="mt-6 p-6 bg-blue-50 border border-blue-200 rounded-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Result</h3>
            <div className="flex items-center justify-center mb-4">
              <div className="text-center">
                <div className="text-5xl font-bold text-blue-600">{result.predicted_rating.toFixed(2)}</div>
                <div className="text-sm text-gray-600 mt-2">Predicted Rating (out of 5)</div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-blue-200">
              <p className="text-sm text-gray-600">
                <span className="font-semibold">Features Used:</span>
                <ul className="list-disc list-inside mt-2 space-y-1">
                  <li>Votes: {result.features_used.votes}</li>
                  <li>Online Order: {result.features_used.online_order ? 'Yes' : 'No'}</li>
                  <li>Book Table: {result.features_used.book_table ? 'Yes' : 'No'}</li>
                  <li>Location: {result.features_used.location}</li>
                  <li>Restaurant Type: {result.features_used.rest_type}</li>
                  <li>Cuisines: {result.features_used.cuisines}</li>
                  <li>Average Cost: {result.features_used.cost}</li>
                </ul>
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default RatingPrediction;