# LLM Topic Clustering Project

This project implements a simple solution for identifying topics in a small dataset and clustering documents thematically using the Mistral-7B model. The focus is on optimizing the implementation for local resources, specifically on an M3 MacBook Air.

## Project Structure

```
llm-topic-clustering
├── data
│   └── sample_dataset.csv        # Contains the small dataset used for topic clustering.
├── src
│   ├── preprocess.py             # Functions for lightweight text preprocessing.
│   ├── embed.py                  # Generates embeddings using the Mistral-7B model.
│   ├── cluster.py                # Implements clustering logic using k-means.
│   ├── label.py                  # Extracts representative words/phrases for labeling clusters.
│   └── main.py                   # Entry point of the application.
├── models
│   └── mistral-7b                # Contains the quantized version of the Mistral-7B model.
├── requirements.txt              # Lists the dependencies required for the project.
├── README.md                     # Documentation for the project.
└── .gitignore                    # Specifies files and directories to ignore in version control.
```

## Dataset

The dataset used for this project is located in `data/sample_dataset.csv`. It consists of short documents or summaries to optimize processing and reduce computational load.

## Technologies Used

- **Python**: The programming language used for implementation.
- **Hugging Face Transformers**: For loading and using the Mistral-7B model.
- **scikit-learn**: For implementing the k-means clustering algorithm.
- **spaCy**: For lightweight text preprocessing.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd llm-topic-clustering
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that the quantized Mistral-7B model is placed in the `models/mistral-7b` directory.

## Usage

To run the project, execute the following command:
```
python src/main.py
```

This will initiate the entire pipeline, including preprocessing, embedding generation, clustering, and labeling.

## Challenges and Solutions

- **Model Size**: The Mistral-7B model is large, but quantization and CPU optimization make it feasible for local execution.
- **Clustering Interpretability**: Clustering is simplified by using a smaller number of clusters (5-10) and manually validating results.

## Efficiency Measurement

The project includes methods for measuring the quality of clustering, including manual inspection of clusters, silhouette scores, and tracking runtime and memory usage during inference and clustering.

## License

This project is licensed under the MIT License.