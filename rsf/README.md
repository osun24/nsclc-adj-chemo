# RSF Tree Visualizer with AI-Powered Features

An interactive Streamlit application for visualizing Random Survival Forest (RSF) decision trees with integrated Kaplan-Meier survival analysis and AI-powered feature descriptions.

## Features

### Core Functionality
- **Tree Navigation**: Browse through individual trees in the RSF model using Next/Back buttons
- **Interactive Node Selection**: Click on nodes to trigger survival analysis
- **Hierarchical Tree Layout**: Root nodes at top, leaves at bottom with automatic zoom-to-fit

### Survival Analysis
- **Split Node Analysis**: Click blue split nodes to view Kaplan-Meier curves comparing patient groups based on feature thresholds
- **Leaf Node Analysis**: Click colored leaf nodes to compare survival outcomes between different leaf nodes
- **Statistical Testing**: Automatic log-rank tests with p-values and number-at-risk tables

### AI-Powered Feature Descriptions
- **Hover Intelligence**: Hover over blue split nodes for 1 second to get AI-generated feature descriptions
- **Clinical Context**: LLM provides contextual information about genomic features in NSCLC survival analysis
- **Cached Responses**: Feature descriptions are cached to improve performance and reduce API calls

## Setup Instructions

### 1. Install Dependencies
```bash
# Run the installation script
./install_ai_features.sh

# Or manually install
pip install openai python-dotenv
```

### 2. Configure OpenAI API
1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Edit the `.env` file:
```env
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Run the Application
```bash
streamlit run rsf/see-forest-app.py
```

## Usage Guide

### Navigation
- Use **Next/Back** buttons to browse through different trees in the forest
- Current tree number is displayed (e.g., "Currently viewing tree 5 of 750")

### Node Interactions

#### Split Nodes (Blue)
- **Click**: Shows feature-based Kaplan-Meier plot comparing patients above/below the split threshold
- **Hover**: After 1 second, displays AI-generated description of the feature's clinical significance

#### Leaf Nodes (Colored)
- **Click**: Shows comparative Kaplan-Meier plot between the selected leaf and other leaves
- **Color**: Represents survival characteristics (red = poor prognosis, black = better prognosis)

#### Root Node (Green)
- **Click**: Shows feature-based KM plot if it's a split node
- **Hover**: Shows feature description if available

### AI Feature Descriptions
When you hover over a blue split node, the system:
1. Shows "🔍 Looking up feature information..." while querying the LLM
2. Displays a clinical description covering:
   - What the feature represents (gene, pathway, clinical variable)
   - Known role in NSCLC prognosis/treatment
   - Impact on patient survival outcomes

## Technical Details

### Architecture
- **Frontend**: Streamlit web interface with PyVis network visualization
- **Visualization**: vis-network library with hierarchical layout
- **Survival Analysis**: lifelines library for Kaplan-Meier estimation
- **AI Integration**: OpenAI GPT-4o-mini for feature descriptions
- **State Management**: URL parameters preserve tree and node selections across page reloads

### Data Flow
1. User hovers over split node → JavaScript detects hover event
2. After 1-second delay → Feature name extracted from node info
3. URL parameter set → Streamlit receives feature request
4. LLM query → Clinical description generated and cached
5. UI updates → Description displayed in right panel

### Performance Optimizations
- **Caching**: Feature descriptions cached in session state
- **Debouncing**: 1-second hover delay prevents excessive API calls
- **Fallback Handling**: Graceful degradation when OpenAI unavailable
- **Auto-fit**: Automatic zoom-to-fit for optimal tree viewing

## File Structure

```
rsf/
├── see-forest-app.py           # Main Streamlit application
├── rsf_results_affy/           # RSF model and feature data
│   ├── *.pkl                   # Trained RSF models
│   └── *_importances_*.csv     # Feature importance rankings
├── .env                        # OpenAI API configuration
└── install_ai_features.sh      # Dependency installation script
```

## Dependencies

### Core Requirements
- streamlit
- numpy
- pandas
- networkx
- pyvis
- matplotlib
- lifelines
- joblib

### AI Features (Optional)
- openai
- python-dotenv

## Troubleshooting

### OpenAI Integration Issues
- **No API key**: Configure `.env` file with valid OpenAI API key
- **Rate limits**: Feature descriptions are cached to minimize API usage
- **Network errors**: App continues to work without AI features if OpenAI is unavailable

### Visualization Issues
- **Tree cutoff**: Increased height to 1000px and auto-fit functionality
- **Node selection**: URL parameters preserve state across page reloads
- **Hover timing**: 1-second delay ensures intentional hover actions

### Performance
- **Large models**: Application tested with 750-tree RSF models
- **Memory usage**: Session state caching optimizes repeated operations
- **Loading time**: Initial model loading may take a few seconds

## Example Use Cases

1. **Clinical Research**: Explore which genomic features drive treatment decisions in NSCLC patients
2. **Biomarker Discovery**: Identify key survival-associated features across different trees
3. **Model Interpretation**: Understand how RSF models make survival predictions
4. **Educational**: Learn about genomic features and their clinical significance through AI descriptions

## Future Enhancements

- Export functionality for survival plots
- Feature importance visualization
- Tree comparison tools
- Enhanced AI prompts for more specific clinical contexts
- Integration with additional survival models
