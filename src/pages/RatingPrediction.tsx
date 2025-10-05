import { useState, useEffect } from 'react';
import { TrendingUp, Loader2, RefreshCw, BarChart3, Award } from 'lucide-react';

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

interface PredictionOptions {
  locations: string[];
  rest_types: string[];
  cuisines: string[];
  random_sample: {
    votes: number;
    online_order: number;
    book_table: number;
    location: string;
    rest_type: string;
    cuisines: string;
    cost: number;
  };
  stats: {
    votes_range: { min: number; max: number; avg: number };
    cost_range: { min: number; max: number; avg: number };
  };
}

interface ModelInsights {
  metrics: {
    train_r2: number;
    test_r2: number;
    train_rmse: number;
    test_rmse: number;
    train_mae: number;
    test_mae: number;
    train_accuracy: number;
    test_accuracy: number;
  };
  feature_importance: Array<{ feature: string; importance: number }>;
}

const Skeleton = ({ className = "" }: { className?: string }) => (
  <div className={`animate-pulse rounded-md bg-gray-200 ${className}`} />
);

const GaugeChart = ({ value, label }: { value: number; label: string }) => {
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;
  
  const getColor = (val: number) => {
    if (val >= 80) return '#10b981';
    if (val >= 60) return '#3b82f6';
    if (val >= 40) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-40 h-40">
        <svg className="transform -rotate-90 w-40 h-40">
          <circle
            cx="80"
            cy="80"
            r={radius}
            stroke="#e5e7eb"
            strokeWidth="12"
            fill="none"
          />
          <circle
            cx="80"
            cy="80"
            r={radius}
            stroke={getColor(value)}
            strokeWidth="12"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold" style={{ color: getColor(value) }}>
            {value.toFixed(1)}%
          </span>
        </div>
      </div>
      <p className="text-sm text-gray-600 mt-2 font-medium">{label}</p>
    </div>
  );
};

const BarChart = ({ data, title }: { data: Array<{ label: string; value: number }>; title: string }) => {
  const maxValue = Math.max(...data.map(d => d.value));
  
  return (
    <div className="bg-white p-4 rounded-lg border border-gray-200">
      <h4 className="text-sm font-semibold text-gray-700 mb-4">{title}</h4>
      <div className="space-y-3">
        {data.map((item, idx) => (
          <div key={idx} className="flex items-center gap-3">
            <span className="text-xs text-gray-600 w-16 text-right">{item.label}</span>
            <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
              <div
                className="bg-gradient-to-r from-blue-500 to-blue-600 h-full rounded-full flex items-center justify-end px-2 transition-all duration-700 ease-out"
                style={{ width: `${(item.value / maxValue) * 100}%` }}
              >
                <span className="text-xs text-white font-medium">{item.value.toFixed(3)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const FeatureImportanceChart = ({ data }: { data: Array<{ feature: string; importance: number }> }) => {
  const sortedData = [...data].sort((a, b) => b.importance - a.importance).slice(0, 7);
  const maxImportance = Math.max(...sortedData.map(d => d.importance));
  
  const colors = [
    'from-purple-500 to-purple-600',
    'from-blue-500 to-blue-600',
    'from-green-500 to-green-600',
    'from-yellow-500 to-yellow-600',
    'from-orange-500 to-orange-600',
    'from-red-500 to-red-600',
    'from-pink-500 to-pink-600',
  ];

  return (
    <div className="bg-white p-4 rounded-lg border border-gray-200">
      <h4 className="text-sm font-semibold text-gray-700 mb-4 flex items-center gap-2">
        <Award className="w-4 h-4" />
        Feature Importance
      </h4>
      <div className="space-y-3">
        {sortedData.map((item, idx) => (
          <div key={idx} className="flex items-center gap-3">
            <span className="text-xs text-gray-600 w-24 text-right truncate" title={item.feature}>
              {item.feature}
            </span>
            <div className="flex-1 bg-gray-100 rounded-full h-7 overflow-hidden">
              <div
                className={`bg-gradient-to-r ${colors[idx % colors.length]} h-full rounded-full flex items-center justify-end px-3 transition-all duration-700 ease-out`}
                style={{ width: `${(item.importance / maxImportance) * 100}%` }}
              >
                <span className="text-xs text-white font-medium">{(item.importance * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

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
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [insightsLoading, setInsightsLoading] = useState(true);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [options, setOptions] = useState<PredictionOptions | null>(null);
  const [insights, setInsights] = useState<ModelInsights | null>(null);
  const [showInsights, setShowInsights] = useState(false);

  const [formData, setFormData] = useState({
    votes: 100,
    online_order: 1,
    book_table: 0,
    location: '',
    rest_type: '',
    cuisines: '',
    cost: 500,
  });

  useEffect(() => {
    fetchOptions();
    fetchInsights();
  }, []);

  const fetchOptions = async () => {
    setOptionsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/predict-rating/options');
      if (response.ok) {
        const data = await response.json();
        setOptions(data);
        setFormData(data.random_sample);
      } else {
        throw new Error('Failed to fetch prediction options.');
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setOptionsLoading(false);
    }
  };

  const fetchInsights = async () => {
    setInsightsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/predict-rating/model-insights');
      if (response.ok) {
        const data = await response.json();
        setInsights(data);
      }
    } catch (err: any) {
      console.error('Failed to fetch insights:', err);
    } finally {
      setInsightsLoading(false);
    }
  };

  const loadRandomSample = () => {
    if (options?.random_sample) {
      fetchOptions();
      setResult(null);
      setSuccess(null);
      setError(null);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setSuccess(null);

    try {
      const payload = {
        ...formData,
        online_order: Number(formData.online_order),
        book_table: Number(formData.book_table),
        votes: Number(formData.votes),
        cost: Number(formData.cost),
      };

      const response = await fetch('http://localhost:8000/api/predict-rating', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
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
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (optionsLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="bg-white shadow-xl rounded-xl p-8 w-full max-w-2xl">
          <div className="flex items-center justify-center mb-6">
            <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
          </div>
          <p className="text-center text-gray-600">Loading prediction options...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white shadow-xl rounded-xl p-8 mb-6">
          <div className="flex items-center justify-center mb-6">
            <TrendingUp className="w-12 h-12 text-blue-500" />
          </div>
          <h2 className="text-3xl font-bold text-center text-gray-800 mb-2">Predict Rating</h2>
          <p className="text-center text-gray-500 mb-4">
            Enter restaurant features to predict its aggregate rating.
          </p>
          
          <div className="flex justify-center gap-4 mb-6">
            <button
              onClick={loadRandomSample}
              className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
              disabled={loading}
            >
              <RefreshCw className="w-4 h-4" />
              Load Random Sample
            </button>
            <button
              onClick={() => setShowInsights(!showInsights)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors"
            >
              <BarChart3 className="w-4 h-4" />
              {showInsights ? 'Hide' : 'Show'} Model Insights
            </button>
          </div>
        </div>

        {/* Model Insights Section */}
        {showInsights && insights && !insightsLoading && (
          <div className="bg-white shadow-xl rounded-xl p-8 mb-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
              <BarChart3 className="w-6 h-6 text-blue-500" />
              Model Performance Insights
            </h3>
            
            {/* Accuracy Gauges */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl">
                <GaugeChart value={insights.metrics.train_accuracy} label="Training Accuracy" />
              </div>
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl">
                <GaugeChart value={insights.metrics.test_accuracy} label="Test Accuracy" />
              </div>
            </div>

            {/* Metrics Charts */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <BarChart
                title="R² Score (Coefficient of Determination)"
                data={[
                  { label: 'Train', value: insights.metrics.train_r2 },
                  { label: 'Test', value: insights.metrics.test_r2 },
                ]}
              />
              <BarChart
                title="RMSE (Root Mean Squared Error)"
                data={[
                  { label: 'Train', value: insights.metrics.train_rmse },
                  { label: 'Test', value: insights.metrics.test_rmse },
                ]}
              />
            </div>

            {/* Feature Importance */}
            <FeatureImportanceChart data={insights.feature_importance} />
          </div>
        )}

        {insightsLoading && showInsights && (
          <div className="bg-white shadow-xl rounded-xl p-8 mb-6">
            <Skeleton className="h-8 w-64 mb-6" />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <Skeleton className="h-48" />
              <Skeleton className="h-48" />
            </div>
          </div>
        )}

        {/* Prediction Form */}
        <div className="bg-white shadow-xl rounded-xl p-8">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                  min={options?.stats.votes_range.min || 0}
                  max={options?.stats.votes_range.max || 10000}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 px-3 py-2 border"
                  required
                />
                <p className="text-xs text-gray-500 mt-1">
                  Range: {options?.stats.votes_range.min} - {options?.stats.votes_range.max}
                </p>
              </div>

              <div>
                <label htmlFor="cost" className="block text-sm font-medium text-gray-700">
                  Average Cost for Two
                </label>
                <input
                  type="number"
                  id="cost"
                  name="cost"
                  value={formData.cost}
                  onChange={handleChange}
                  min={options?.stats.cost_range.min || 0}
                  max={options?.stats.cost_range.max || 10000}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 px-3 py-2 border"
                  required
                />
                <p className="text-xs text-gray-500 mt-1">
                  Range: {options?.stats.cost_range.min} - {options?.stats.cost_range.max}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="online_order" className="block text-sm font-medium text-gray-700">
                  Online Order
                </label>
                <select
                  id="online_order"
                  name="online_order"
                  value={formData.online_order.toString()}
                  onChange={handleChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 px-3 py-2 border"
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
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 px-3 py-2 border"
                >
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>
            </div>

            <div>
              <label htmlFor="location" className="block text-sm font-medium text-gray-700">
                Location
              </label>
              <select
                id="location"
                name="location"
                value={formData.location}
                onChange={handleChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 px-3 py-2 border"
                required
              >
                {options?.locations.map((loc, index) => (
                  <option key={index} value={loc}>{loc}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="rest_type" className="block text-sm font-medium text-gray-700">
                Restaurant Type
              </label>
              <select
                id="rest_type"
                name="rest_type"
                value={formData.rest_type}
                onChange={handleChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 px-3 py-2 border"
                required
              >
                {options?.rest_types.map((type, index) => (
                  <option key={index} value={type}>{type}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="cuisines" className="block text-sm font-medium text-gray-700">
                Cuisines
              </label>
              <select
                id="cuisines"
                name="cuisines"
                value={formData.cuisines}
                onChange={handleChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 px-3 py-2 border"
                required
              >
                {options?.cuisines.map((cuisine, index) => (
                  <option key={index} value={cuisine}>{cuisine}</option>
                ))}
              </select>
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
                <div className="text-sm text-gray-600">
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
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default RatingPrediction;