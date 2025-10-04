import { useState, useEffect } from 'react';
import { ThumbsUp, Loader2, RefreshCw, Filter, Star, DollarSign, Users, MapPin, Utensils, Clock, Sparkles, TrendingUp } from 'lucide-react';

interface Restaurant {
  name: string;
  cuisine: string;
  city: string;
  rating: number;
  cost: number;
  votes: number;
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
  const [showFilters, setShowFilters] = useState(false);

  const [filters, setFilters] = useState({
    cuisine: '',
    location: '',
    count: 10,
    minRating: 0,
    maxCost: 5000,
    minCost: 0,
    hasOnlineDelivery: 'any',
    hasTableBooking: 'any',
    sortBy: 'similarity'
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
        }),
      });

      if (!response.ok) throw new Error('Recommendation failed');

      const data = await response.json();
      let filteredRecs = data.recommendations || [];

      if (filters.minRating > 0) {
        filteredRecs = filteredRecs.filter((r: Restaurant) => r.rating >= filters.minRating);
      }
      if (filters.maxCost < 5000) {
        filteredRecs = filteredRecs.filter((r: Restaurant) => r.cost <= filters.maxCost);
      }
      if (filters.minCost > 0) {
        filteredRecs = filteredRecs.filter((r: Restaurant) => r.cost >= filters.minCost);
      }
      if (filters.hasOnlineDelivery !== 'any') {
        const hasDelivery = filters.hasOnlineDelivery === 'yes';
        filteredRecs = filteredRecs.filter((r: Restaurant) => 
          (r.online_delivery?.toLowerCase() === 'yes') === hasDelivery
        );
      }
      if (filters.hasTableBooking !== 'any') {
        const hasBooking = filters.hasTableBooking === 'yes';
        filteredRecs = filteredRecs.filter((r: Restaurant) => 
          (r.table_booking?.toLowerCase() === 'yes') === hasBooking
        );
      }

      if (filters.sortBy === 'rating') {
        filteredRecs.sort((a: Restaurant, b: Restaurant) => b.rating - a.rating);
      } else if (filters.sortBy === 'cost_low') {
        filteredRecs.sort((a: Restaurant, b: Restaurant) => a.cost - b.cost);
      } else if (filters.sortBy === 'cost_high') {
        filteredRecs.sort((a: Restaurant, b: Restaurant) => b.cost - a.cost);
      } else if (filters.sortBy === 'popularity') {
        filteredRecs.sort((a: Restaurant, b: Restaurant) => b.votes - a.votes);
      }

      setRecommendations(filteredRecs);
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
      minRating: 0,
      maxCost: 5000,
      minCost: 0,
      hasOnlineDelivery: 'any',
      hasTableBooking: 'any',
      sortBy: 'similarity'
    });
    setRecommendations([]);
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
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Sparkles className="h-8 w-8 text-blue-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Restaurant Recommendations</h1>
            <p className="text-sm text-gray-500 mt-1">Find your perfect dining experience</p>
          </div>
        </div>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors"
        >
          <Filter className="w-4 h-4" />
          {showFilters ? 'Hide' : 'Show'} Advanced Filters
        </button>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
                {options?.cities.map((city) => (
                  <option key={city} value={city}>{city}</option>
                ))}
              </select>
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
          </div>

          {showFilters && (
            <div className="pt-6 border-t border-gray-200 space-y-6">
              <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                <Filter className="w-5 h-5 mr-2 text-blue-600" />
                Advanced Filters
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                    <Star className="w-4 h-4 mr-2 text-yellow-500" />
                    Minimum Rating: {filters.minRating.toFixed(1)} ⭐
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="5"
                    step="0.5"
                    value={filters.minRating}
                    onChange={(e) => setFilters(prev => ({ ...prev, minRating: Number(e.target.value) }))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Any</span>
                    <span>Excellent</span>
                  </div>
                </div>

                <div>
                  <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                    <DollarSign className="w-4 h-4 mr-2 text-green-500" />
                    Maximum Cost: ₹{filters.maxCost}
                  </label>
                  <input
                    type="range"
                    min="100"
                    max="5000"
                    step="100"
                    value={filters.maxCost}
                    onChange={(e) => setFilters(prev => ({ ...prev, maxCost: Number(e.target.value) }))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>₹100</span>
                    <span>₹5000+</span>
                  </div>
                </div>

                <div>
                  <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                    <Clock className="w-4 h-4 mr-2 text-purple-500" />
                    Online Delivery
                  </label>
                  <select
                    value={filters.hasOnlineDelivery}
                    onChange={(e) => setFilters(prev => ({ ...prev, hasOnlineDelivery: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="any">Any</option>
                    <option value="yes">Available</option>
                    <option value="no">Not Required</option>
                  </select>
                </div>

                <div>
                  <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                    <Users className="w-4 h-4 mr-2 text-orange-500" />
                    Table Booking
                  </label>
                  <select
                    value={filters.hasTableBooking}
                    onChange={(e) => setFilters(prev => ({ ...prev, hasTableBooking: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="any">Any</option>
                    <option value="yes">Available</option>
                    <option value="no">Not Required</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
                  <TrendingUp className="w-4 h-4 mr-2 text-blue-600" />
                  Sort Results By
                </label>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                  {[
                    { value: 'similarity', label: 'Best Match' },
                    { value: 'rating', label: 'Highest Rated' },
                    { value: 'cost_low', label: 'Price: Low to High' },
                    { value: 'cost_high', label: 'Price: High to Low' },
                    { value: 'popularity', label: 'Most Popular' },
                  ].map((sort) => (
                    <button
                      key={sort.value}
                      onClick={() => setFilters(prev => ({ ...prev, sortBy: sort.value }))}
                      className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                        filters.sortBy === sort.value
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      {sort.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

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
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        )}
      </div>

      {recommendations.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">
              Found {recommendations.length} Perfect Matches
            </h2>
            <div className="text-sm text-gray-500">
              Sorted by: <span className="font-semibold text-blue-600">
                {filters.sortBy === 'similarity' ? 'Best Match' :
                 filters.sortBy === 'rating' ? 'Highest Rated' :
                 filters.sortBy === 'cost_low' ? 'Price (Low)' :
                 filters.sortBy === 'cost_high' ? 'Price (High)' : 'Most Popular'}
              </span>
            </div>
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
                  <div className="flex gap-2">
                    {restaurant.online_delivery?.toLowerCase() === 'yes' && (
                      <span className="bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded-full font-medium">
                        🚚 Delivery
                      </span>
                    )}
                    {restaurant.table_booking?.toLowerCase() === 'yes' && (
                      <span className="bg-orange-100 text-orange-700 text-xs px-2 py-1 rounded-full font-medium">
                        📅 Booking
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-500">
                    {restaurant.votes} votes
                  </div>
                </div>

                {restaurant.similarity_score && (
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