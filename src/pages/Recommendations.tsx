import { useState, useEffect } from 'react';
import { ThumbsUp, Loader2, Star } from 'lucide-react';
import { Skeleton } from "@/components/ui/skeleton";

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

const RecommendationSkeleton = () => (
  <div className="mt-6">
    <Skeleton className="h-8 w-1/2 mb-4" />
    <div className="space-y-4">
      <Skeleton className="h-24 w-full rounded-lg" />
      <Skeleton className="h-24 w-full rounded-lg" />
      <Skeleton className="h-24 w-full rounded-lg" />
    </div>
  </div>
);

function Recommendations() {
  const [loading, setLoading] = useState(false);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [result, setResult] = useState<RecommendationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cuisines, setCuisines] = useState<string[]>([]);
  const [cities, setCities] = useState<string[]>([]);

  // Corrected state to match backend's model
  const [formData, setFormData] = useState({
    cuisine: '',
    location: '', 
    count: 5,
  });

  useEffect(() => {
    fetchOptions();
  }, []);

  const fetchOptions = async () => {
    setOptionsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/recommend-restaurants/options');
      if (response.ok) {
        const data = await response.json();
        setCuisines(data.cuisines || []);
        setCities(data.cities || []);
        setFormData(prev => ({
          ...prev,
          cuisine: data.cuisines[0] || '',
          location: data.cities[0] || '',
        }));
      } else {
        throw new Error('Failed to fetch recommendation options.');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setOptionsLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ 
      ...prev, 
      [name]: name === 'count' ? parseInt(value, 10) : value 
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/recommend-restaurants', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
        body: JSON.stringify(formData),
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
        throw new Error(`Recommendation failed with status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
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
          <ThumbsUp className="w-12 h-12 text-green-500" />
        </div>
        <h2 className="text-3xl font-bold text-center text-gray-800 mb-2">Get Recommendations</h2>
        <p className="text-center text-gray-500 mb-8">
          Find the best restaurants based on your preferences.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="city" className="block text-sm font-medium text-gray-700">
              City
            </label>
            <select
              id="city"
              name="location"
              value={formData.location}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500"
              required
            >
              {optionsLoading ? (
                <option>Loading cities...</option>
              ) : (
                cities.map((city, index) => <option key={index} value={city}>{city}</option>)
              )}
            </select>
          </div>

          <div>
            <label htmlFor="cuisine" className="block text-sm font-medium text-gray-700">
              Cuisine
            </label>
            <select
              id="cuisine"
              name="cuisine"
              value={formData.cuisine}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500"
              required
            >
              {optionsLoading ? (
                <option>Loading cuisines...</option>
              ) : (
                cuisines.map((cuisine, index) => <option key={index} value={cuisine}>{cuisine}</option>)
              )}
            </select>
          </div>
          <button
            type="submit"
            className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
              loading || optionsLoading ? 'bg-green-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500'
            }`}
            disabled={loading || optionsLoading}
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Finding...
              </>
            ) : (
              'Get Recommendations'
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
        
        {loading && <RecommendationSkeleton />}

        {result && result.recommendations.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Top {result.recommendations.length} Recommendations
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