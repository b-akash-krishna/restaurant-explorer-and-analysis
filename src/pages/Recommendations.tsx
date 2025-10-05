import { useState, useEffect } from 'react';
import { ThumbsUp, Loader2, RefreshCw, Star, DollarSign, Users, MapPin, Utensils, CheckCircle, XCircle, Sparkles, TrendingUp } from 'lucide-react';

interface Restaurant {
  name: string;
  cuisine: string;
  city: string;
  rating: number;
  cost: number;
  votes: number;
  price_range: number;
  online_delivery: string;
  table_booking: string;
  similarity_score: number;
}

interface RecommendationOptions {
  cuisines: string[];
  cities: string[];
}

function Recommendations() {
  const [loading, setLoading] = useState(false);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [recommendations, setRecommendations] = useState<Restaurant[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [options, setOptions] = useState<RecommendationOptions | null>(null);

  const [filters, setFilters] = useState({
    cuisine: '',
    location: '',
    count: 10,
    price_range: null as number | null,
    table_booking: null as boolean | null,
    online_delivery: null as boolean | null,
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
        setOptions(data);
        if (data.cuisines?.length > 0) {
          setFilters(prev => ({ ...prev, cuisine: data.cuisines[0] }));
        }
        if (data.cities?.length > 0) {
          setFilters(prev => ({ ...prev, location: data.cities[0] }));
        }
      }
    } catch (err) {
      setError('Failed to load options');
    } finally {
      setOptionsLoading(false);
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setRecommendations([]);

    try {
      const response = await fetch('http://localhost:8000/api/recommend-restaurants', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cuisine: filters.cuisine,
          location: filters.location,
          count: filters.count,
          price_range: filters.price_range,
          table_booking: filters.table_booking,
          online_delivery: filters.online_delivery,
        }),
      });

      if (!response.ok) throw new Error('Recommendation failed');

      const data = await response.json();
      setRecommendations(data.recommendations || []);
    } catch (err) {
      setError('Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  const resetFilters = () => {
    setFilters({
      cuisine: options?.cuisines[0] || '',
      location: options?.cities[0] || '',
      count: 10,
      price_range: null,
      table_booking: null,
      online_delivery: null,
    });
    setRecommendations([]);
  };

  const getPriceRangeLabel = (range: number) => {
    const labels = ['', 'Budget (₹)', 'Moderate (₹₹)', 'Expensive (₹₹₹)', 'Very Expensive (₹₹₹₹)'];
    return labels[range] || 'Unknown';
  };

  if (optionsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="animate-spin h-8 w-8 text-blue-600" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <Sparkles className="h-8 w-8 text-blue-600" />
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Restaurant Recommendations</h1>
          <p className="text-sm text-gray-500 mt-1">Find your perfect dining experience with personalized filters</p>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <Users className="h-5 w-5 mr-2 text-blue-600" />
          Set Your Preferences
        </h2>

        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                <MapPin className="w-4 h-4 mr-2 text-blue-600" />
                City / Location
              </label>
              <select
                value={filters.location}
                onChange={(e) => setFilters(prev => ({ ...prev, location: e.target.value }))}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {options?.cities.map((city) => (
                  <option key={city} value={city}>{city}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                <Utensils className="w-4 h-4 mr-2 text-blue-600" />
                Cuisine Type
              </label>
              <select
                value={filters.cuisine}
                onChange={(e) => setFilters(prev => ({ ...prev, cuisine: e.target.value }))}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {options?.cuisines.map((cuisine) => (
                  <option key={cuisine} value={cuisine}>{cuisine}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                <DollarSign className="w-4 h-4 mr-2 text-green-600" />
                Price Range
              </label>
              <select
                value={filters.price_range === null ? '' : filters.price_range}
                onChange={(e) => setFilters(prev => ({ 
                  ...prev, 
                  price_range: e.target.value === '' ? null : Number(e.target.value) 
                }))}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Any Price</option>
                <option value="1">Budget (₹)</option>
                <option value="2">Moderate (₹₹)</option>
                <option value="3">Expensive (₹₹₹)</option>
                <option value="4">Very Expensive (₹₹₹₹)</option>
              </select>
            </div>

            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                <CheckCircle className="w-4 h-4 mr-2 text-purple-600" />
                Table Booking
              </label>
              <select
                value={filters.table_booking === null ? '' : String(filters.table_booking)}
                onChange={(e) => setFilters(prev => ({ 
                  ...prev, 
                  table_booking: e.target.value === '' ? null : e.target.value === 'true' 
                }))}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Any</option>
                <option value="true">Available</option>
                <option value="false">Not Required</option>
              </select>
            </div>

            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                <TrendingUp className="w-4 h-4 mr-2 text-orange-600" />
                Online Delivery
              </label>
              <select
                value={filters.online_delivery === null ? '' : String(filters.online_delivery)}
                onChange={(e) => setFilters(prev => ({ 
                  ...prev, 
                  online_delivery: e.target.value === '' ? null : e.target.value === 'true' 
                }))}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Any</option>
                <option value="true">Available</option>
                <option value="false">Not Required</option>
              </select>
            </div>
          </div>

          <div>
            <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
              <Users className="w-4 h-4 mr-2 text-blue-600" />
              Number of Results
            </label>
            <select
              value={filters.count}
              onChange={(e) => setFilters(prev => ({ ...prev, count: Number(e.target.value) }))}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="5">5 restaurants</option>
              <option value="10">10 restaurants</option>
              <option value="15">15 restaurants</option>
              <option value="20">20 restaurants</option>
            </select>
          </div>

          <div className="flex gap-3">
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="flex-1 bg-blue-600 text-white py-3 rounded-md font-medium hover:bg-blue-700 transition-colors disabled:bg-blue-300 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin h-5 w-5 mr-2" />
                  Finding Restaurants...
                </>
              ) : (
                <>
                  <ThumbsUp className="h-5 w-5 mr-2" />
                  Get Recommendations
                </>
              )}
            </button>
            <button
              onClick={resetFilters}
              className="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md font-medium transition-colors flex items-center"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Reset
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        )}
      </div>

      {recommendations.length > 0 && (
        <>
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg shadow-md p-6 border-2 border-blue-200">
            <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center">
              <Sparkles className="h-6 w-6 mr-2 text-blue-600" />
              User Preference Testing - Top Recommendations
            </h2>
            <p className="text-sm text-gray-600 mb-4">
              Based on your preferences: <span className="font-semibold">{filters.cuisine}</span> in <span className="font-semibold">{filters.location}</span>
              {filters.price_range && <>, {getPriceRangeLabel(filters.price_range)}</>}
              {filters.table_booking !== null && <>, Table Booking: {filters.table_booking ? 'Required' : 'Not Required'}</>}
              {filters.online_delivery !== null && <>, Online Delivery: {filters.online_delivery ? 'Required' : 'Not Required'}</>}
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recommendations.slice(0, 6).map((restaurant, index) => (
                <div
                  key={index}
                  className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-all border-l-4 border-blue-500"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <h3 className="font-bold text-base text-gray-900 mb-1 line-clamp-1">
                        {restaurant.name}
                      </h3>
                      <p className="text-xs text-gray-600 flex items-center">
                        <Utensils className="w-3 h-3 mr-1" />
                        {restaurant.cuisine.split(',')[0]}
                      </p>
                    </div>
                    <div className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full font-bold text-xs flex items-center shrink-0">
                      <Star className="w-3 h-3 mr-1 fill-yellow-500" />
                      {restaurant.rating.toFixed(1)}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center text-xs text-gray-600">
                      <MapPin className="w-3 h-3 mr-1 text-blue-600" />
                      <span className="font-medium">{restaurant.city}</span>
                    </div>

                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center text-green-700 font-semibold">
                        <DollarSign className="w-3 h-3 mr-1" />
                        ₹{restaurant.cost}
                      </div>
                      <div className="text-gray-500">
                        {restaurant.votes} votes
                      </div>
                    </div>

                    <div className="flex gap-1 flex-wrap">
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        restaurant.price_range <= 2 ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-700'
                      }`}>
                        {getPriceRangeLabel(restaurant.price_range)}
                      </span>
                      {restaurant.online_delivery?.toLowerCase() === 'yes' && (
                        <span className="bg-purple-100 text-purple-700 text-xs px-2 py-0.5 rounded-full">
                          🚚 Delivery
                        </span>
                      )}
                      {restaurant.table_booking?.toLowerCase() === 'yes' && (
                        <span className="bg-blue-100 text-blue-700 text-xs px-2 py-0.5 rounded-full">
                          📅 Booking
                        </span>
                      )}
                    </div>

                    {restaurant.similarity_score > 0 && (
                      <div className="pt-2 border-t border-gray-200">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-gray-600">Match</span>
                          <span className="font-semibold text-blue-600">
                            {(restaurant.similarity_score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-purple-500 h-1 rounded-full"
                            style={{ width: `${restaurant.similarity_score * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">
                All {recommendations.length} Recommendations
              </h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recommendations.map((restaurant, index) => (
                <div
                  key={index}
                  className="border border-gray-200 rounded-lg p-5 hover:shadow-lg transition-all hover:border-blue-300 bg-gradient-to-br from-white to-gray-50"
                >
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex-1">
                      <h3 className="font-bold text-lg text-gray-900 mb-1">{restaurant.name}</h3>
                      <p className="text-sm text-gray-600 flex items-center">
                        <Utensils className="w-3 h-3 mr-1" />
                        {restaurant.cuisine}
                      </p>
                    </div>
                    <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full font-bold text-sm flex items-center">
                      <Star className="w-4 h-4 mr-1 fill-yellow-500" />
                      {restaurant.rating.toFixed(1)}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div className="bg-blue-50 p-2 rounded">
                      <div className="text-xs text-gray-600">Location</div>
                      <div className="font-semibold text-sm text-blue-700 flex items-center">
                        <MapPin className="w-3 h-3 mr-1" />
                        {restaurant.city}
                      </div>
                    </div>
                    <div className="bg-green-50 p-2 rounded">
                      <div className="text-xs text-gray-600">Cost for Two</div>
                      <div className="font-semibold text-sm text-green-700 flex items-center">
                        <DollarSign className="w-3 h-3 mr-1" />
                        ₹{restaurant.cost}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-3 border-t border-gray-200">
                    <div className="flex gap-2 flex-wrap">
                      <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                        restaurant.price_range <= 2 ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-700'
                      }`}>
                        {getPriceRangeLabel(restaurant.price_range)}
                      </span>
                      {restaurant.online_delivery?.toLowerCase() === 'yes' ? (
                        <span className="bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded-full font-medium flex items-center">
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Delivery
                        </span>
                      ) : (
                        <span className="bg-gray-100 text-gray-600 text-xs px-2 py-1 rounded-full font-medium flex items-center">
                          <XCircle className="w-3 h-3 mr-1" />
                          No Delivery
                        </span>
                      )}
                      {restaurant.table_booking?.toLowerCase() === 'yes' ? (
                        <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded-full font-medium flex items-center">
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Booking
                        </span>
                      ) : (
                        <span className="bg-gray-100 text-gray-600 text-xs px-2 py-1 rounded-full font-medium flex items-center">
                          <XCircle className="w-3 h-3 mr-1" />
                          No Booking
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500">
                      {restaurant.votes} votes
                    </div>
                  </div>

                  {restaurant.similarity_score > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-200">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-600">Match Score</span>
                        <span className="font-semibold text-blue-600">
                          {(restaurant.similarity_score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-purple-500 h-1.5 rounded-full"
                          style={{ width: `${restaurant.similarity_score * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {!loading && recommendations.length === 0 && !error && (
        <div className="bg-white rounded-lg shadow-md p-12 text-center">
          <Sparkles className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-700 mb-2">
            No recommendations yet
          </h3>
          <p className="text-gray-500">
            Set your preferences and click "Get Recommendations" to find your perfect restaurant
          </p>
        </div>
      )}
    </div>
  );
}

export default Recommendations;