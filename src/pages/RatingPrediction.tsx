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
    <div className="bg-white/95 backdrop-blur-sm p-4 rounded-lg border border-gray-200 shadow-sm">
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
    <div className="bg-white/95 backdrop-blur-sm p-4 rounded-lg border border-gray-200 shadow-sm">
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
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center p-4">
        <div className="bg-white/95 backdrop-blur-md shadow-2xl rounded-2xl p-8 w-full max-w-2xl border border-blue-200">
          <div className="flex items-center justify-center mb-6">
            <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
          </div>
          <p className="text-center text-gray-600">Loading prediction options...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Hero Header */}
      <div 
        className="relative bg-cover bg-center py-16 mb-8"
        style={{
          backgroundImage: "url('https://images.unsplash.com/photo-1559339352-11d035aa65de?w=1600&h=400&fit=crop')",
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900/90 via-blue-800/85 to-purple-900/90"></div>
        <div className="relative max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-white/20 backdrop-blur-md p-4 rounded-2xl">
              <TrendingUp className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-center text-white mb-3 drop-shadow-lg">
            Restaurant Rating Prediction
          </h1>
          <p className="text-center text-blue-100 text-lg max-w-2xl mx-auto drop-shadow-md">
            Leverage machine learning to predict restaurant ratings based on key features and attributes
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 pb-8">
        {/* Control Panel */}
        <div className="bg-white/95 backdrop-blur-sm shadow-xl rounded-2xl p-6 mb-6 border border-blue-100">
          <div className="flex justify-center gap-4">
            <button
              onClick={loadRandomSample}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-gray-100 to-gray-200 hover:from-gray-200 hover:to-gray-300 text-gray-700 rounded-xl transition-all duration-300 font-medium shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
              disabled={loading}
            >
              <RefreshCw className="w-4 h-4" />
              Load Random Sample
            </button>
            <button
              onClick={() => setShowInsights(!showInsights)}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-xl transition-all duration-300 font-medium shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
            >
              <BarChart3 className="w-4 h-4" />
              {showInsights ? 'Hide' : 'Show'} Model Insights
            </button>
          </div>
        </div>

        {/* Model Insights Section */}
        {showInsights && insights && !insightsLoading && (
          <div className="bg-white/95 backdrop-blur-sm shadow-xl rounded-2xl p-8 mb-6 border border-blue-100">
            <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
              <BarChart3 className="w-6 h-6 text-blue-600" />
              Model Performance Insights
            </h3>
            
            {/* Accuracy Gauges */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-2xl shadow-inner">
                <GaugeChart value={insights.metrics.train_accuracy} label="Training Accuracy" />
              </div>
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-2xl shadow-inner">
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
          <div className="bg-white/95 backdrop-blur-sm shadow-xl rounded-2xl p-8 mb-6 border border-blue-100">
            <Skeleton className="h-8 w-64 mb-6" />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <Skeleton className="h-48" />
              <Skeleton className="h-48" />
            </div>
          </div>
        )}

        {/* Prediction Form */}
        <div className="bg-white/95 backdrop-blur-sm shadow-xl rounded-2xl p-8 border border-blue-100">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="votes" className="block text-sm font-semibold text-gray-700 mb-2">
                  Number of Votes
                </label>
                <input
                  type="number"
                  id="votes"
                  name="votes"
                  value={formData.votes}
                  onChange={handleChange}
                  min={options?.stats.votes_range.min || 0}
                  max={options?.stats.votes_range.max || 10000}
                  className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-200"
                  required
                />
                <p className="text-xs text-gray-500 mt-2">
                  Range: {options?.stats.votes_range.min} - {options?.stats.votes_range.max}
                </p>
              </div>

              <div>
                <label htmlFor="cost" className="block text-sm font-semibold text-gray-700 mb-2">
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
                  className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-200"
                  required
                />
                <p className="text-xs text-gray-500 mt-2">
                  Range: {options?.stats.cost_range.min} - {options?.stats.cost_range.max}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="online_order" className="block text-sm font-semibold text-gray-700 mb-2">
                  Online Order Available
                </label>
                <select
                  id="online_order"
                  name="online_order"
                  value={formData.online_order.toString()}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-200"
                >
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>

              <div>
                <label htmlFor="book_table" className="block text-sm font-semibold text-gray-700 mb-2">
                  Table Booking Available
                </label>
                <select
                  id="book_table"
                  name="book_table"
                  value={formData.book_table.toString()}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-200"
                >
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>
            </div>

            <div>
              <label htmlFor="location" className="block text-sm font-semibold text-gray-700 mb-2">
                Location
              </label>
              <select
                id="location"
                name="location"
                value={formData.location}
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-200"
                required
              >
                {options?.locations.map((loc, index) => (
                  <option key={index} value={loc}>{loc}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="rest_type" className="block text-sm font-semibold text-gray-700 mb-2">
                Restaurant Type
              </label>
              <select
                id="rest_type"
                name="rest_type"
                value={formData.rest_type}
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-200"
                required
              >
                {options?.rest_types.map((type, index) => (
                  <option key={index} value={type}>{type}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="cuisines" className="block text-sm font-semibold text-gray-700 mb-2">
                Cuisines
              </label>
              <select
                id="cuisines"
                name="cuisines"
                value={formData.cuisines}
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-200"
                required
              >
                {options?.cuisines.map((cuisine, index) => (
                  <option key={index} value={cuisine}>{cuisine}</option>
                ))}
              </select>
            </div>

            <button
              type="submit"
              className={`w-full flex justify-center items-center py-4 px-6 rounded-xl shadow-lg text-base font-semibold text-white transition-all duration-300 transform hover:-translate-y-1 ${
                loading ? 'bg-blue-400 cursor-not-allowed' : 'bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 hover:shadow-xl'
              }`}
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Processing Prediction...
                </>
              ) : (
                'Predict Rating'
              )}
            </button>
          </form>

          {error && (
            <div className="mt-6 p-4 bg-red-50 border-2 border-red-200 rounded-xl">
              <p className="text-red-600 text-sm">
                <span className="font-bold">Error:</span> {error}
              </p>
            </div>
          )}

          {success && (
            <div className="mt-6 p-4 bg-green-50 border-2 border-green-200 rounded-xl">
              <p className="text-green-600 text-sm">
                <span className="font-bold">Success:</span> {success}
              </p>
            </div>
          )}
          
          {loading && !result && <ResultSkeleton />}

          {!loading && result && (
            <div className="mt-6 p-8 bg-gradient-to-br from-blue-50 to-purple-50 border-2 border-blue-200 rounded-2xl shadow-lg">
              <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                <TrendingUp className="w-6 h-6 text-blue-600" />
                Prediction Result
              </h3>
              <div className="flex items-center justify-center mb-6">
                <div className="text-center bg-white/80 backdrop-blur-sm p-8 rounded-2xl shadow-md">
                  <div className="text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
                    {result.predicted_rating.toFixed(2)}
                  </div>
                  <div className="text-sm text-gray-600 font-medium">Predicted Rating (out of 5)</div>
                </div>
              </div>
              <div className="mt-6 pt-6 border-t-2 border-blue-200">
                <div className="text-sm text-gray-700 bg-white/60 backdrop-blur-sm p-4 rounded-xl">
                  <span className="font-bold text-gray-900">Features Used:</span>
                  <ul className="list-disc list-inside mt-3 space-y-2">
                    <li>Votes: <span className="font-semibold">{result.features_used.votes}</span></li>
                    <li>Online Order: <span className="font-semibold">{result.features_used.online_order ? 'Yes' : 'No'}</span></li>
                    <li>Book Table: <span className="font-semibold">{result.features_used.book_table ? 'Yes' : 'No'}</span></li>
                    <li>Location: <span className="font-semibold">{result.features_used.location}</span></li>
                    <li>Restaurant Type: <span className="font-semibold">{result.features_used.rest_type}</span></li>
                    <li>Cuisines: <span className="font-semibold">{result.features_used.cuisines}</span></li>
                    <li>Average Cost: <span className="font-semibold">₹{result.features_used.cost}</span></li>
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