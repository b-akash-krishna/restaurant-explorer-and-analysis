import { useState, useEffect } from 'react';
import { ThumbsUp, Loader2, Star } from 'lucide-react';

interface Restaurant {
  name: string;
  cuisine: string;
  city: string;
  rating: number;
  price_range: number;
  votes: number;
}

interface RecommendationResult {
  recommendations: Restaurant[];
  count: number;
}

function Recommendations() {
  const [loading, setLoading] = useState(false);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [result, setResult] = useState<RecommendationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cuisines, setCuisines] = useState<string[]>([]);
  const [cities, setCities] = useState<string[]>([]);

  const [formData, setFormData] = useState({
    cuisine: '',
    city: '',
    price_range: 0,
    top_n: 10,
  });

  useEffect(() => {
    fetchOptions();
  }, []);

  const fetchOptions = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/recommend-restaurants/options');
      if (response.ok) {
        const data = await response.json();
        setCuisines(data.cuisines || []);
        setCities(data.cities || []);
      }
    } catch (err) {
      console.error('Failed to fetch options');
    } finally {
      setOptionsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/recommend-restaurants', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cuisine: formData.cuisine || null,
          city: formData.city || null,
          price_range: formData.price_range || null,
          top_n: formData.top_n,
        }),
      });

      if (!response.ok) throw new Error('Recommendation failed');

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Failed to get recommendations. Make sure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <ThumbsUp className="h-8 w-8 text-green-600" />
        <h1 className="text-3xl font-bold text-gray-900">Restaurant Recommendations</h1>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <p className="text-gray-600 mb-6">
          Get personalized restaurant recommendations based on your preferences for cuisine, location, and price range.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Preferred Cuisine
              </label>
              <select
                value={formData.cuisine}
                onChange={(e) => setFormData({ ...formData, cuisine: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 focus:border-transparent"
                disabled={optionsLoading}
              >
                <option value="">Any Cuisine</option>
                {cuisines.slice(0, 30).map((cuisine) => (
                  <option key={cuisine} value={cuisine}>
                    {cuisine}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                City
              </label>
              <select
                value={formData.city}
                onChange={(e) => setFormData({ ...formData, city: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 focus:border-transparent"
                disabled={optionsLoading}
              >
                <option value="">Any City</option>
                {cities.slice(0, 30).map((city) => (
                  <option key={city} value={city}>
                    {city}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Price Range
              </label>
              <select
                value={formData.price_range}
                onChange={(e) => setFormData({ ...formData, price_range: parseInt(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 focus:border-transparent"
              >
                <option value="0">Any Price</option>
                <option value="1">1 - Budget</option>
                <option value="2">2 - Moderate</option>
                <option value="3">3 - Expensive</option>
                <option value="4">4 - Very Expensive</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Number of Recommendations
              </label>
              <input
                type="number"
                value={formData.top_n}
                onChange={(e) => setFormData({ ...formData, top_n: parseInt(e.target.value) })}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 focus:border-transparent"
                min="1"
                max="50"
                required
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-green-600 text-white py-3 rounded-md font-medium hover:bg-green-700 transition-colors disabled:bg-green-300 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin h-5 w-5 mr-2" />
                Finding Recommendations...
              </>
            ) : (
              'Get Recommendations'
            )}
          </button>
        </form>

        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        )}

        {result && result.recommendations.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Found {result.count} Recommendations
            </h3>
            <div className="grid grid-cols-1 gap-4">
              {result.recommendations.map((restaurant, index) => (
                <div
                  key={index}
                  className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900 text-lg">{restaurant.name}</h4>
                      <p className="text-sm text-gray-600 mt-1">{restaurant.cuisine}</p>
                      <p className="text-sm text-gray-500 mt-1">{restaurant.city}</p>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center space-x-1">
                        <Star className="h-5 w-5 text-yellow-500 fill-current" />
                        <span className="font-semibold text-gray-900">{restaurant.rating}</span>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">{restaurant.votes} votes</p>
                      <p className="text-xs text-gray-500 mt-1">
                        Price: {'$'.repeat(restaurant.price_range)}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Recommendations;
