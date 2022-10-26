{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "1) Import the necessary libraries"
      ],
      "metadata": {
        "id": "bOpIXNUq4n3D"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "GhT5RUqq4ZWF"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from keras import utils\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from keras.models import Model\n",
        "from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding\n",
        "from keras.optimizers import RMSprop\n",
        "from keras.preprocessing.text import Tokenizer\n",
        "from keras.preprocessing import sequence\n",
        "from keras.utils import to_categorical\n",
        "%matplotlib inline"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')\n",
        "df"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        },
        "id": "hRolXJbS4ub_",
        "outputId": "5f4b7e88-0e40-4eeb-b2ad-23d72535e6b2"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        v1                                                 v2 Unnamed: 2  \\\n",
              "0      ham  Go until jurong point, crazy.. Available only ...        NaN   \n",
              "1      ham                      Ok lar... Joking wif u oni...        NaN   \n",
              "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN   \n",
              "3      ham  U dun say so early hor... U c already then say...        NaN   \n",
              "4      ham  Nah I don't think he goes to usf, he lives aro...        NaN   \n",
              "...    ...                                                ...        ...   \n",
              "5567  spam  This is the 2nd time we have tried 2 contact u...        NaN   \n",
              "5568   ham              Will Ì_ b going to esplanade fr home?        NaN   \n",
              "5569   ham  Pity, * was in mood for that. So...any other s...        NaN   \n",
              "5570   ham  The guy did some bitching but I acted like i'd...        NaN   \n",
              "5571   ham                         Rofl. Its true to its name        NaN   \n",
              "\n",
              "     Unnamed: 3 Unnamed: 4  \n",
              "0           NaN        NaN  \n",
              "1           NaN        NaN  \n",
              "2           NaN        NaN  \n",
              "3           NaN        NaN  \n",
              "4           NaN        NaN  \n",
              "...         ...        ...  \n",
              "5567        NaN        NaN  \n",
              "5568        NaN        NaN  \n",
              "5569        NaN        NaN  \n",
              "5570        NaN        NaN  \n",
              "5571        NaN        NaN  \n",
              "\n",
              "[5572 rows x 5 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-f10af594-dcc0-4884-ba73-a90a5ced6877\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>v1</th>\n",
              "      <th>v2</th>\n",
              "      <th>Unnamed: 2</th>\n",
              "      <th>Unnamed: 3</th>\n",
              "      <th>Unnamed: 4</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>ham</td>\n",
              "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>ham</td>\n",
              "      <td>Ok lar... Joking wif u oni...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>spam</td>\n",
              "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>ham</td>\n",
              "      <td>U dun say so early hor... U c already then say...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>ham</td>\n",
              "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5567</th>\n",
              "      <td>spam</td>\n",
              "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5568</th>\n",
              "      <td>ham</td>\n",
              "      <td>Will Ì_ b going to esplanade fr home?</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5569</th>\n",
              "      <td>ham</td>\n",
              "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5570</th>\n",
              "      <td>ham</td>\n",
              "      <td>The guy did some bitching but I acted like i'd...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5571</th>\n",
              "      <td>ham</td>\n",
              "      <td>Rofl. Its true to its name</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5572 rows × 5 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f10af594-dcc0-4884-ba73-a90a5ced6877')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f10af594-dcc0-4884-ba73-a90a5ced6877 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f10af594-dcc0-4884-ba73-a90a5ced6877');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "2)Preprocessing"
      ],
      "metadata": {
        "id": "4Qi2eBLV45xk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)\n",
        "df"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        },
        "id": "ZEnFpXy34vHh",
        "outputId": "af59fac7-dc48-4998-afd9-506d16de3f9a"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        v1                                                 v2\n",
              "0      ham  Go until jurong point, crazy.. Available only ...\n",
              "1      ham                      Ok lar... Joking wif u oni...\n",
              "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
              "3      ham  U dun say so early hor... U c already then say...\n",
              "4      ham  Nah I don't think he goes to usf, he lives aro...\n",
              "...    ...                                                ...\n",
              "5567  spam  This is the 2nd time we have tried 2 contact u...\n",
              "5568   ham              Will Ì_ b going to esplanade fr home?\n",
              "5569   ham  Pity, * was in mood for that. So...any other s...\n",
              "5570   ham  The guy did some bitching but I acted like i'd...\n",
              "5571   ham                         Rofl. Its true to its name\n",
              "\n",
              "[5572 rows x 2 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e9b3b2ef-a43a-4faf-a19a-6e1b0af621cb\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>v1</th>\n",
              "      <th>v2</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>ham</td>\n",
              "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>ham</td>\n",
              "      <td>Ok lar... Joking wif u oni...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>spam</td>\n",
              "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>ham</td>\n",
              "      <td>U dun say so early hor... U c already then say...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>ham</td>\n",
              "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5567</th>\n",
              "      <td>spam</td>\n",
              "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5568</th>\n",
              "      <td>ham</td>\n",
              "      <td>Will Ì_ b going to esplanade fr home?</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5569</th>\n",
              "      <td>ham</td>\n",
              "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5570</th>\n",
              "      <td>ham</td>\n",
              "      <td>The guy did some bitching but I acted like i'd...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5571</th>\n",
              "      <td>ham</td>\n",
              "      <td>Rofl. Its true to its name</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5572 rows × 2 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e9b3b2ef-a43a-4faf-a19a-6e1b0af621cb')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-e9b3b2ef-a43a-4faf-a19a-6e1b0af621cb button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-e9b3b2ef-a43a-4faf-a19a-6e1b0af621cb');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(df.v1,palette='Set3')\n",
        "plt.xlabel('Label')\n",
        "plt.title('Number of ham and spam messages')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 367
        },
        "id": "-Z5DXScq49cP",
        "outputId": "2c56e38a-fdaa-4735-fe37-e6b812062385"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  FutureWarning\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Number of ham and spam messages')"
            ]
          },
          "metadata": {},
          "execution_count": 4
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X = df.v2\n",
        "Y = df.v1\n",
        "le = LabelEncoder()\n",
        "Y = le.fit_transform(Y)\n",
        "Y = Y.reshape(-1,1)"
      ],
      "metadata": {
        "id": "AJK7iUhg5C0r"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "3)Split into training and test data."
      ],
      "metadata": {
        "id": "y2CfZr_C5IX6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)"
      ],
      "metadata": {
        "id": "gWapYn985F0T"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "max_words = 1000\n",
        "max_len = 150\n",
        "tok = Tokenizer(num_words=max_words)\n",
        "tok.fit_on_texts(X_train)\n",
        "sequences = tok.texts_to_sequences(X_train)\n",
        "sequences_matrix = utils.pad_sequences(sequences,maxlen=max_len)"
      ],
      "metadata": {
        "id": "B1FWeG9L5LjN"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sequences_matrix.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KZKmTfL-5Obm",
        "outputId": "99f63e10-afd6-42d7-ba17-11358e7dc2de"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(4736, 150)"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sequences_matrix.ndim"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yiLaf6Xl5Q_B",
        "outputId": "70313ec6-7c11-489a-fd62-ab7e14ba7a29"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sequences_matrix = np.reshape(sequences_matrix,(4736,150,1))"
      ],
      "metadata": {
        "id": "RPUn8HzX5S5q"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sequences_matrix.ndim #3d shape verification to proceed to RNN LSTM"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "81SkgZoh5Vla",
        "outputId": "c5c28ecb-163a-4515-d587-19e02531e0c1"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "3"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "4)Create model for RNN"
      ],
      "metadata": {
        "id": "4-Al58g75bDc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from keras.models import Sequential\n",
        "from keras.layers import Dense\n",
        "from keras.layers import LSTM\n",
        "from keras.layers import Embedding\n"
      ],
      "metadata": {
        "id": "mzrzeGfl5ZOG"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = Sequential()"
      ],
      "metadata": {
        "id": "x-Qa4DeO5fW9"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "5)Add Layers"
      ],
      "metadata": {
        "id": "pWUSYNGw5jWP"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.add(Embedding(max_words,50,input_length=max_len))"
      ],
      "metadata": {
        "id": "EsezMng65hYC"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.add(LSTM(units=64,input_shape = (sequences_matrix.shape[1],1),return_sequences=True))"
      ],
      "metadata": {
        "id": "eUBk-8565nNb"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.add(LSTM(units=64,return_sequences=True))"
      ],
      "metadata": {
        "id": "PRup-FdK5p8r"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.add(LSTM(units=64,return_sequences=True))"
      ],
      "metadata": {
        "id": "6zvJHFy55r1Z"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.add(LSTM(units=64))"
      ],
      "metadata": {
        "id": "5X_cN1Mp5ttV"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.add(Dense(units = 256,activation = 'relu'))"
      ],
      "metadata": {
        "id": "G2J0fHxS5vk6"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.add(Dense(units = 1,activation = 'sigmoid'))"
      ],
      "metadata": {
        "id": "o2dnpiWG5xp2"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "6)Compile the model"
      ],
      "metadata": {
        "id": "UjTUFKmS52s7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.summary()\n",
        "model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DK610DWw5zhd",
        "outputId": "2169de65-2362-4162-d749-7bb781328719"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " embedding (Embedding)       (None, 150, 50)           50000     \n",
            "                                                                 \n",
            " lstm (LSTM)                 (None, 150, 64)           29440     \n",
            "                                                                 \n",
            " lstm_1 (LSTM)               (None, 150, 64)           33024     \n",
            "                                                                 \n",
            " lstm_2 (LSTM)               (None, 150, 64)           33024     \n",
            "                                                                 \n",
            " lstm_3 (LSTM)               (None, 64)                33024     \n",
            "                                                                 \n",
            " dense (Dense)               (None, 256)               16640     \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 1)                 257       \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 195,409\n",
            "Trainable params: 195,409\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "M = model.fit(sequences_matrix,Y_train,batch_size=128,epochs=5,validation_split=0.2)\n",
        "M"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aP1MKaNe56gB",
        "outputId": "72f92b88-e64b-484e-d6dc-8be7c41875f4"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/5\n",
            "30/30 [==============================] - 44s 1s/step - loss: 0.4427 - accuracy: 0.8635 - val_loss: 0.3336 - val_accuracy: 0.8713\n",
            "Epoch 2/5\n",
            "30/30 [==============================] - 33s 1s/step - loss: 0.1965 - accuracy: 0.9300 - val_loss: 0.0995 - val_accuracy: 0.9736\n",
            "Epoch 3/5\n",
            "30/30 [==============================] - 33s 1s/step - loss: 0.0634 - accuracy: 0.9820 - val_loss: 0.0891 - val_accuracy: 0.9778\n",
            "Epoch 4/5\n",
            "30/30 [==============================] - 31s 1s/step - loss: 0.0406 - accuracy: 0.9897 - val_loss: 0.0745 - val_accuracy: 0.9821\n",
            "Epoch 5/5\n",
            "30/30 [==============================] - 32s 1s/step - loss: 0.0283 - accuracy: 0.9923 - val_loss: 0.0800 - val_accuracy: 0.9800\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7fbfd235d590>"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "8)Save the model"
      ],
      "metadata": {
        "id": "pNtTJS-m6DL1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.save"
      ],
      "metadata": {
        "id": "DviewH1e59vx",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ff84074b-2c6b-494f-9299-76d65bd10394"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<bound method Model.save of <keras.engine.sequential.Sequential object at 0x7fbfd6a37210>>"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_sequences = tok.texts_to_sequences(X_test)\n",
        "test_sequences_matrix = utils.pad_sequences(test_sequences,maxlen=max_len)"
      ],
      "metadata": {
        "id": "K-ngxrzl6LYl"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accr = model.evaluate(test_sequences_matrix,Y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PGqYRQJ_6Or1",
        "outputId": "a3d955fd-020b-441f-ba97-240406145d32"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "27/27 [==============================] - 5s 95ms/step - loss: 0.1078 - accuracy: 0.9737\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "l = accr[0]\n",
        "a =accr[1]\n",
        "print('Test set\\n  Loss: {:0.3f}\\n  Accuracy: {:0.3f}'.format(l,a))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TIYoAxgY6QJG",
        "outputId": "c36bf59a-ce7e-4b40-89d5-a28a8c512a88"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Test set\n",
            "  Loss: 0.108\n",
            "  Accuracy: 0.974\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Accuracy and Loss Graph"
      ],
      "metadata": {
        "id": "RbgJwjXC6jze"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "results = pd.DataFrame({\"Train Loss\": M.history['loss'], \"Validation Loss\": M.history['val_loss'],\n",
        "              \"Train Accuracy\": M.history['accuracy'], \"Validation Accuracy\": M.history['val_accuracy']\n",
        "             })\n",
        "fig, ax = plt.subplots(nrows=2, figsize=(16, 9))\n",
        "results[[\"Train Loss\", \"Validation Loss\"]].plot(ax=ax[0])\n",
        "results[[\"Train Accuracy\", \"Validation Accuracy\"]].plot(ax=ax[1])\n",
        "ax[0].set_xlabel(\"Epoch\")\n",
        "ax[1].set_xlabel(\"Epoch\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 506
        },
        "id": "3s02IQQA6TF7",
        "outputId": "12bfc460-ab47-434f-c770-3960135303c3"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1152x648 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA6gAAAIWCAYAAABa9qCZAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdeXhU1f3H8feZmSzsSwiEfRMEEvZNQPYtRYW6b1VRq9WqqHVpXVkUtP1R91pt3epSqdqKoEIQZBUFAigm7CBLgLBKAoQsM3N+f0zAJCQhZLuT5PN6njwzc+fOne/kdko+nnPP11hrEREREREREXGay+kCREREREREREABVURERERERIKEAqqIiIiIiIgEBQVUERERERERCQoKqCIiIiIiIhIUFFBFREREREQkKHicLiCvBg0a2FatWjldhoiIiIiIiJSB1atXH7LWRub3XNAF1FatWhEfH+90GSIiIiIiIlIGjDE7C3pOU3xFREREREQkKCigioiIiIiISFBQQBUREREREZGgEHTXoIqIiIiIiOSVlZVFUlIS6enpTpciRRQeHk6zZs0ICQkp8msUUEVEREREJOglJSVRq1YtWrVqhTHG6XLkLKy1HD58mKSkJFq3bl3k12mKr4iIiIiIBL309HQiIiIUTisIYwwRERHnPOKtgHqO9qWcJC3T63QZIiIiIiJVjsJpxVKc86WAeo4e+OgHhv91MZ+v24u11ulyRERERESkHBw+fJhu3brRrVs3oqKiaNq06enHmZmZhb42Pj6eCRMmnNP7tWrVikOHDpWk5ApJ16Ceo/tHtmfiZ4nc/e+1fNBmF5PGRnN+VC2nyxIRERERkTIUERHB999/D8CkSZOoWbMmDz744OnnvV4vHk/+8apXr1706tWrXOqs6DSCeo56t6rP7Hsu5Klfx7B+XypjXlrK5NmJpJzMcro0EREREREpR+PHj+eOO+6gb9++PPzww6xcuZJ+/frRvXt3+vfvz6ZNmwBYtGgRF198MRAIt7fccgtDhgyhTZs2vPTSS0V+vx07djBs2DC6dOnC8OHD2bVrFwAff/wxMTExdO3alUGDBgGQmJhInz596NatG126dGHLli2l/OnLhkZQi8HtMtxwQUsu6tyY6fM28c7yHcz+YS8Px3bgih7NcLk0N15EREREpKxMnp3I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hviKiIiIiIiIX1BAFREREREREb+ggCoiIiIiIiJ+QQFVRERERERE/IICqoiIiIiIiPgFBVQREZEqYozxGGNWlfp6tArP3doYs6aqziciIuKPdB9UERGRqpNnrT3X6SJERERqKnVQRUREqpkxZpsx5m/GmB+MManGmHbF61sbYxYbY1YbYxYZY+KL1zc1xsw0xnxf/HVx8alcxpj/M8asNcZ8boyp59iLEhERqQYKqCIiIlWnXpkhvkNLbTtkre0GvAa8XLzuH8B71truwAfAq8XrXwW+tNb2AM4D1havbw+8bq3tChwEBlfz6xERETmrjLXW6RpERERqBWNMjrU2vJz124C+1tqtxpggYLe1NsoYsx9obq0tKl6/y1obbYzZB8RaawtKnaM1sMBa27748R+AIGvts9X/ykRERM4OdVBFRETODnuC5cooKLXsQXNJiIhILaOAMisodAAAIABJREFUKiIicnYMLfX9m+LlZcCw4uWRwNLi5UXAPQDGGJcxpuHZKlJERMRJ+uRVRESk6tQzxqwq9XietfborWYaG2NW4+uCDi9edz/wjjHmd8A+4Lbi9Q8Ck4wxv8bXKb0H2FXt1YuIiDhM16CKiIhUs+JrUBOttfudrkVERMSfaYiviIiIiIiI+AV1UEVERERERMQvqIMqIiIiIiIifkEBVURERERERPyCAqqIiIiIiIj4BQVUERERERER8QsKqCIiIiIiIuIXFFBFRERERETELyigioiIiIiIiF9QQBURERERERG/oIAqIiIiIiIifkEBVURERERERPyCAqqIiIiIiIj4hUCnCygrOjratm7d2ukyREREREREpBqsWLFiv7U2prxtfhdQW7duTVpamtNliIiIiIiISDUwxmw/0TYN8RURERERERG/oIAqIiIiIiIifkEBVURERERERPyC312DWp6ioiIyMzPJz893uhTxI6GhocTGxhIUFOR0KSIiIiIiUgVqREDNzMwkIiKC1q1bY4xxuhzxA9ZaDhw4QGZmJgkJCU6XIyIiIiIiVaBGDPHNz88nKipK4VRKGGOIiopSV11EREREpBapEQEVUDiV4+h3QkRERESkdqkRQ3xFRERERETqKmstbq/F7bEUeb0Uub24vZYij9e3zuOlyGNxe33f4xrXo0mDUKfLPi0KqBVw4MABrrzySgB2796Ny+UiJiYGgNTU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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    }
  ]
}