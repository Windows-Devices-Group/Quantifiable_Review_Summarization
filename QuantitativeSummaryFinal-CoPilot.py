#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Required Libraries
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import openai
import pyodbc
import urllib
from sqlalchemy import create_engine
import pandas as pd
from azure.identity import InteractiveBrowserCredential
from pandasai import SmartDataframe
import pandas as pd
from pandasai.llm import AzureOpenAI
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import base64
import pandasql as ps
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#Initializing API Keys to use LLM
os.environ["AZURE_OPENAI_API_KEY"] = "a22e367d483f4718b9e96b1f52ce6d53"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://hulk-openai.openai.azure.com/"

#Reading the dataset
Sentiment_Data  = pd.read_csv("New_CoPilot_Data.csv")


# In[2]:


def Sentiment_Score_Derivation(value):
    try:
        if value == "positive":
            return 1
        elif value == "negative":
            return -1
        else:
            return 0
    except Exception as e:
        err = f"An error occurred while deriving Sentiment Score: {e}"
        return err    

#Deriving Sentiment Score and Review Count columns into the dataset
Sentiment_Data["Sentiment_Score"] = Sentiment_Data["Sentiment"].apply(Sentiment_Score_Derivation)
Sentiment_Data["Review_Count"] = 1.0


# In[3]:


def convert_top_to_limit(sql):
    try:
        tokens = sql.upper().split()
        is_top_used = False

        for i, token in enumerate(tokens):
            if token == 'TOP':
                is_top_used = True
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    limit_value = tokens[i + 1]
                    # Remove TOP and insert LIMIT and value at the end
                    del tokens[i:i + 2]
                    tokens.insert(len(tokens), 'LIMIT')
                    tokens.insert(len(tokens), limit_value)
                    break  # Exit loop after successful conversion
                else:
                    raise ValueError("TOP operator should be followed by a number")

        return ' '.join(tokens) if is_top_used else sql
    except Exception as e:
        err = f"An error occurred while converting Top to Limit in SQL Query: {e}"
        return err


# In[4]:


def process_tablename(sql, table_name):
    try:
        x = sql.upper()
        query = x.replace(table_name.upper(), table_name)
        return query
    except Exception as e:
        err = f"An error occurred while processing table name in SQL query: {e}"
        return err


# In[6]:


def get_conversational_chain_quant(history):
    try:
        hist = """"""
        for i in history:
            hist = hist+"\nUser: "+i[0]
            if isinstance(i[1],pd.DataFrame):
                x = i[1].to_string()
            else:
                x = i[1]
            hist = hist+"\nResponse: "+x
        prompt_template = """
        
        If an user is asking for Summarize reviews of any product. Note that user is not seeking for reviews, user is seeking for all the Quantitative things of the product(Net Sentiment & Review Count) and also (Aspect wise sentiment and Aspect wise review count)
        So choose to Provide Net Sentiment and Review Count and Aspect wise sentiment and their respective review count and Union them in single table
        
        Example : If the user Quesiton is "Summarize reviews of CoPilot Produt"
        
        User seeks for net sentiment and aspect wise net sentiment of "Windows 10" Product and their respective review count in a single table
        
        Your response should be : Overall Sentiment is nothing but the net sentiment and overall review count of the product
        
                        Aspect Aspect_SENTIMENT REVIEW_COUNT
                    0 TOTAL 40 15000.0
                    1 Generic 31.8 2302.0
                    2 Microsoft Product 20.2 570.0
                    3 Productivity 58.9 397.0
                    4 Code Generation -1.2 345.0
                    5 Ease of Use 20.1 288.0
                    6 Interface -22.9 271.0
                    7 Connectivity -43.7 247.0
                    8 Compatibility -28.6 185.0
                    9 Innovation 52.9 170.0
                    10 Text Summarization/Generation 19.1 157.0
                    11 Reliability -44.7 152.0
                    12 Price 29.5 95.0
                    13 Customization/Personalization 18.9 90.0
                    14 Security/Privacy -41.3 75.0
                    15 Accessibility 16.7 6.0
                    
                    The Query has to be like this 
                    
                SELECT 'TOTAL' AS Aspect, 
                ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                SUM(Review_Count) AS Review_Count
                FROM Sentiment_Data
                WHERE Product_Family LIKE '%CoPilot for Mobile%'

                UNION

                SELECT Aspect, 
                ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                SUM(Review_Count) AS Review_Count
                FROM Sentiment_Data
                WHERE Product_Family LIKE '%CoPilot for Mobile%'
                GROUP BY Aspect

                ORDER BY Review_Count DESC

                    
                    
                IMPORTANT : if any particular Aspect "Code Generation" in user prompt:
                    

                        SELECT 'TOTAL' AS Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'

                        UNION

                        SELECT Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'
                        GROUP BY Aspect
                        HAVING Aspect LIKE %'Code Generation'%

                        ORDER BY Review_Count DESC
                
                
            This is aspect wise summary. If a user wants in Geography level 
        
        
            SELECT Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Net Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Sentiment_Data
                        WHERE Product_Family LIKE '%Asus Rog Zephyrus%'
                        GROUP BY Geography

                        ORDER BY Review_Count DESC
                        
            You shold respond like this. Same Goes for all the segregation


        
        IMPORTANT : IT has to be Net sentiment and Aspect Sentiment. Create 2 SQL Query and UNION them
        
        1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
            2. There is only one table with table name Sentiment_Data where each row is a user review. The table has 10 columns, they are:
                Review: Review of the Copilot Product
                Data_Source: From where is the review taken. It contains different retailers
                Geography: From which Country or Region the review was given. It contains different Grography.
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review. It contains following values: "Windows 11 (Preinstall)", "Windows 10"
                Product_Family: Which version or type of the corresponding Product was the review posted for. Different Device Names
                Sentiment: What is the sentiment of the review. It contains following values: 'Positive', 'Neutral', 'Negative'.
                Aspect: The review is talking about which aspect or feature of the product. It contains following values: "Audio-Microphone","Software","Performance","Storage/Memory","Keyboard","Browser","Connectivity","Hardware","Display","Graphics","Battery","Gaming","Design","Ports","Price","Camera","Customer-Service","Touchpad","Account","Generic"
                Keyword: What are the keywords mentioned in the product
                Review_Count - It will be 1 for each review or each row
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
                
            3. Sentiment mark is calculated by sum of Sentiment_Score.
            4. Net sentiment is calculcated by sum of Sentiment_Score divided by sum of Review_Count. It should be in percentage. Example:
                    SELECT ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment 
                    FROM Sentiment_Data
                    ORDER BY Net_Sentiment DESC
            5. Net sentiment across country or across region is sentiment mark of a country divided by total reviews of that country. It should be in percentage.
                Example to calculate net sentiment across country:
                    SELECT Geography, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Sentiment_Data
                    GROUP BY Geography
                    ORDER BY Net_Sentiment DESC
            6. Net Sentiment across a column "X" is calculcated by Sentiment Mark for each "X" divided by Total Reviews for each "X".
                Example to calculate net sentiment across a column "X":
                    SELECT X, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Sentiment_Data
                    GROUP BY X
                    ORDER BY Net_Sentiment DESC
            7. Distribution of sentiment is calculated by sum of Review_Count for each Sentiment divided by overall sum of Review_Count
                Example: 
                    SELECT Sentiment, SUM(ReviewCount)*100/(SELECT SUM(Review_Count) AS Reviews FROM Sentiment_Data) AS Total_Reviews 
                    FROM Sentiment_Data 
                    GROUP BY Sentiment
                    ORDER BY Total_Reviews DESC
            8. Convert numerical outputs to float upto 1 decimal point.
            9. Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
            10. Top Country is based on Sentiment_Score i.e., the Country which have highest sum(Sentiment_Score)
            11. Always use 'LIKE' operator whenever they mention about any Country. Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
            12. If you are using any field in the aggregate function in select statement, make sure you add them in GROUP BY Clause.
            13. Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS.
            14. Important: Always show Net_Sentiment in Percentage upto 1 decimal point. Hence always make use of ROUND function while giving out Net Sentiment and Add % Symbol after it.
            15. Important: User can ask question about any categories including Aspects, Geograpgy, Sentiment etc etc. Hence, include the in SQL Query if someone ask it.
            16. Important: You Response should directly starts from SQL query nothing else.
            17. Important: Always use LIKE keyword instead of = symbol while generating SQL query.
            18. Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
            19. Sort all Quantifiable outcomes based on review count
        \n Following is the previous conversation from User and Response, use it to get context only:""" + hist + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2023-12-01-preview',
            temperature = 0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization: {e}"
        return err

#Function to convert user prompt to quantitative outputs for Copilot Review Summarization
def query_quant(user_question, history, vector_store_path="faiss_index_CopilotSample"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant(history)
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        SQL_Query = convert_top_to_limit(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Sentiment_Data")
#         print(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except Exception as e:
        err = f"An error occurred while generating response for quantitative review summarization: {e}"
        return err


# In[50]:


def get_conversational_chain_detailed_summary(history):
    try:
        hist = """"""
        for i in history:
            hist = hist+"\nUser: "+i[0]
            if isinstance(i[1],pd.DataFrame):
                x = i[1].to_string()
            else:
                x = i[1]
            hist = hist+"\nResponse: "+ x
        prompt_template = """
        
        
    
        
        1. Your Job is to analyse the Net Sentiment, Aspect wise sentiment and Key word regarding the different aspect and summarize the reviews that user asks for utilizing the reviews and numbers you get. Use maximum use of the numbers and Justify the numbers using the reviews.
        
        ASPECT WISE:
        
                Your will receive Aspect wise net sentiment of the Product. you have to concentrate on top 4 Aspects.
                For that top 4 Aspect you will get top 2 keywords for each aspect. You will receive each keywords' contribution and +ve mention % and negative mention %
                You will receive reviews of that devices focused on these aspects and keywords.

                For Each Aspect

                Condition 1 : If the net sentiment is less than aspect sentiment, which means that particular aspect is driving the net sentiment Higher for that Product. In this case provide why the aspect sentiment is lower than net sentiment.
                Condition 2 : If the net sentiment is high than aspect sentiment, which means that particular aspect is driving the net sentiment Lower for that Product. In this case provide why the aspect sentiment is higher than net sentiment. 

                    IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.

                    Your summary should justify the above conditions and tie in with the net sentiment and aspect sentiment and keywords. Mention the difference between Net Sentiment and Aspect Sentiment (e.g., -2% or +2% higher than net sentiment) in your summary and provide justification.

                    Your response should be : 

                    For Each Aspect 
                            Net Sentiment of the Product and aspect sentiment of that aspect of the Product (Mention Code Generation, Aspect Sentiment) . 
                            Top Keyword contribution and their positive and negative percentages and summarize Reviews what user have spoken regarding this keywords in 2 to 3 lines detailed
                            Top 2nd Keyword contribution and their positive and negative percentages and summarize Reviews what user have spoken regarding this keywords in 2 to 3 lines detailed
                               Limit yourself to top 3 keywords and don't mention as top 1, top 2, top 3 and all. Mention them as pointers
                            Overall Summary

                    IMPORTANT : Example Template :

                    ALWAYS FOLLOW THIS TEMPLATE : Don't miss any of the below:

                    Response : "BOLD ALL THE NUMBERS"

                    IMPOPRTANT : Start with : "These are the 4 major aspects users commented about" and mention their review count contributions

                                   These are the 4 major aspects users commented about:

                                - Total Review for CoPilot for Mobile Product is 1200
                                - Code Generarion: 13.82% of the reviews mentioned this aspect
                                - Ease of Use: 11.08% of the reviews mentioned this aspect
                                - Compatibility: 9.71% of the reviews mentioned this aspect
                                - Interface: 7.37% of the reviews mentioned this aspect

                                Code Generation:
                                - The aspect sentiment for price is 52.8%, which is higher than the net sentiment of 38.5%. This indicates that the aspect of price is driving the net sentiment higher for the Vivobook.
                                -  The top keyword for price is "buy" with a contribution of 28.07%. It has a positive percentage of 13.44% and a negative percentage of 4.48%.
                                      - Users mentioned that the Vivobook offers good value for the price and is inexpensive.
                                - Another top keyword for price is "price" with a contribution of 26.89%. It has a positive percentage of 23.35% and a negative percentage of 0.24%.
                                    - Users praised the affordable price of the Vivobook and mentioned that it is worth the money.

                                Ease of use:
                                - The aspect sentiment for performance is 36.5%, which is lower than the net sentiment of 38.5%. This indicates that the aspect of performance is driving the net sentiment lower for the Vivobook.
                                - The top keyword for performance is "fast" with a contribution of 18.24%. It has a positive percentage of 16.76% and a neutral percentage of 1.47%.
                                    - Users mentioned that the Vivobook is fast and offers good speed.
                                - Another top keyword for performance is "speed" with a contribution of 12.06%. It has a positive percentage of 9.12% and a negative percentage of 2.06%.
                                    - Users praised the speed of the Vivobook and mentioned that it is efficient.


                                lIKE THE ABOVE ONE EXPLAIN OTHER 2 ASPECTS

                                Overall Summary:
                                The net sentiment for the Vivobook is 38.5%, while the aspect sentiment for price is 52.8%, performance is 36.5%, software is 32.2%, and design is 61.9%. This indicates that the aspects of price and design are driving the net sentiment higher, while the aspects of performance and software are driving the net sentiment lower for the Vivobook. Users mentioned that the Vivobook offers good value for the price, is fast and efficient in performance, easy to set up and use in terms of software, and has a sleek and high-quality design.

                                Some Pros and Cons of the device, 


                   IMPORTANT : Do not ever change the above template of Response. Give Spaces accordingly in the response to make it more readable.

                   A Good Response should contains all the above mentioned poniters in the example. 
                       1. Net Sentiment and The Aspect Sentiment
                       2. Total % of mentions regarding the Aspect
                       3. A Quick Summary of whether the aspect is driving the sentiment high or low
                       4. Top Keyword: "Usable" (Contribution: 33.22%, Positive: 68.42%, Negative: 6.32%)
                            - Users have praised the usable experience on the Cobilot for Mobile, with many mentioning the smooth usage and easy to use
                            - Some users have reported experiencing lag while not very great to use, but overall, the gaming Ease of use is highly rated.

                        Top 3 Keywords : Their Contribution, Postitive mention % and Negative mention % and one ot two positive mentions regarding this keywords in each pointer

                        5. IMPORTANT : Pros and Cons in pointers (overall, not related to any aspect)
                        6. Overall Summary
                
           IMPORTANT : If you receive Geography wise sentiment or Region wise sentiment:
           
           
                   If user mentioned Questions like this : Summarize reviews of CoPilot for Mobile across differnet Geography?.

                    Summarize all the reviews across Geography in pointer along with its net sentiment and Review count. 

                    Example : 

                    US : Net Sentiment and Review count

                    Summary in 3 lines

                    CA : Net Sentiment and Review count

                    Summary in 3 lines

                    and so on.

                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.
            
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.\n Following is the previous conversation from User and Response, use it to get context only:""" + hist + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2023-12-01-preview',
            temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_detailed_summary(user_question, history, vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed_summary(history)
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err


# In[51]:


def get_conversational_chain_detailed(history):
    try:
        hist = """"""
        for i in history:
            hist = hist+"\nUser: "+i[0]
            if isinstance(i[1],pd.DataFrame):
                x = i[1].to_string()
            else:
                x = i[1]
            hist = hist+"\nResponse: "+ x
        prompt_template = """
        
        1. Your Job is to analyse the Net Sentiment Aspect wise sentiment and Key word regarding the aspect and summarize the reviews that user asks for utilizing the reviews and numbers you get. Use maximum use of the numbers and Justify the numbers using the reviews.
        
        Overall Sentiment is the Net Sentiment.
        
        Condition 1 : If the net sentiment is less than aspect sentiment, which means that particular aspect is driving the net sentiment Higher for that Product. In this case provide why the aspect sentiment is lower than net sentiment.
        Condition 2 : If the net sentiment is high than aspect sentiment, which means that particular aspect is driving the net sentiment Lower for that Product. In this case provide why the aspect sentiment is higher than net sentiment.
            
            You must be receiving keywords information. If there are any keywords which have more keyword_contribution mention that keyword with its contribution percentage and Positive, negative percentages. 
            Give the reviews summarized for this aspect 
            
            Give at least top 2 keyword information - (Contribution , Positive and Negative Percentage) and when summarizing reviews focus on those particular keywords.
            

            IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.

            Your summary should justify the above conditions and tie in with the net sentiment and aspect sentiment and keywords. Mention the difference between Net Sentiment and Aspect Sentiment (e.g., -2% or +2% higher than net sentiment) in your summary and provide justification.
            
            
            IMPORTANT : Example Template :
            
            ALWAYS FOLLOW THIS TEMPLATE : Don't miss any of the below: 1st Template
                        
                        
            Response : "BOLD ALL THE NUMBERS"
            
            
                    Net Sentiment: 41.9%
                    Aspect Sentiment (Interface): 53.1%

                    75% of the users commented about Interface of this Product. Interface drives the sentiment high for CoPilot for Mobile Product

                    Top Keyword: User-Friendly (Contribution: 33.22%, Positive: 68.42%, Negative: 6.32%)
                    - Users have praised the User-Friendly experience on the CoPilot for Mobile, with many mentioning the good layout and interfacce
                    - Some users have reported experiencing lag while gaming, but overall, the gaming performance is highly rated.

                    Top 2nd Keyword: Graphical (Contribution: 33.22%, Positive: 60%, Negative: 8.42%)
                    - Users appreciate the ability to play various games on the Lenovo Legion, mentioning the enjoyable gaming experience.
                    - A few users have mentioned encountering some issues with certain games, but the majority have had a positive experience.

                    Top 3rd Keyword: Play (Contribution: 16.08%, Positive: 56.52%, Negative: 13.04%)
                    - Users mention the ease of playing games on the Lenovo Legion, highlighting the smooth gameplay and enjoyable experience.
                    - Some users have reported difficulties with certain games, experiencing lag or other performance issues.

                    Pros:
                    1. Smooth gameplay experience
                    2. High FPS and enjoyable gaming performance
                    3. Wide range of games available
                    4. Positive feedback on gaming experience
                    5. Ease of playing games

                    Cons:
                    1. Some users have reported lag or performance issues while gaming
                    2. Occasional difficulties with certain games

                    Overall Summary:
                    The net sentiment for the CoPilot for Mobile is 41.9%, while the aspect sentiment for Inteface is 53.1%. This indicates that the Interface aspect is driving the net sentiment higher for the product. Users have praised the smooth gameplay, high FPS, and enjoyable gaming experience on the Lenovo Legion. The top keywords related to gaming contribute significantly to the aspect sentiment, with positive percentages ranging from 56.52% to 68.42%. However, there are some reports of lag and performance issues with certain games. Overall, the Lenovo Legion is highly regarded for its gaming capabilities, but there is room for improvement in addressing performance issues for a seamless gaming experience.
               
           IMPORTANT : Do not ever change the above template of Response. Give Spaces accordingly in the response to make it more readable.
           
           A Good Response should contains all the above mentioned poniters in the example. 
               1. Net Sentiment and The Aspect Sentiment
               2. Total % of mentions regarding the Aspect
               3. A Quick Summary of whether the aspect is driving the sentiment high or low
               4. Top Keyword: Gaming (Contribution: 33.22%, Positive: 68.42%, Negative: 6.32%)
                    - Users have praised the gaming experience on the Lenovo Legion, with many mentioning the smooth gameplay and high FPS.
                    - Some users have reported experiencing lag while gaming, but overall, the gaming performance is highly rated.
                    
                Top 3 Keywords : Their Contribution, Postitive mention % and Negative mention % and one ot two positive mentions regarding this keywords in each pointer
                
                5. Pros and Cons in pointers
                6. Overall Summary. 
                
        IMPORTANT : Only follow this template. Donot miss out any poniters from the above template

                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.
            
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.\n Following is the previous conversation from User and Response, use it to get context only:""" + hist + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2023-12-01-preview',
            temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_detailed(user_question, history, vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed(history)
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err


# In[52]:


# import numpy as np

# def custom_color_gradient(val, vmin, vmax):
#     green_hex = '#347c47'
#     middle_hex = '#dcdcdc'
#     lower_hex = '#b0343c'
    
#     # Adjust the normalization to set the middle value as 0
#     try:
#         normalized_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
#     except ZeroDivisionError:
#         normalized_val = 0.5
    
#     normalized_val = (normalized_val - 0.5) * 2  # Scale and shift to set middle value as 0
    
#     if normalized_val <= 0:
#         # Interpolate between lower_hex and middle_hex for values <= 0
#         r = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[1:3], 16), int(middle_hex[1:3], 16)]))
#         g = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[3:5], 16), int(middle_hex[3:5], 16)]))
#         b = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[5:7], 16), int(middle_hex[5:7], 16)]))
#     else:
#         # Interpolate between middle_hex and green_hex for values > 0
#         r = int(np.interp(normalized_val, [0, 1], [int(middle_hex[1:3], 16), int(green_hex[1:3], 16)]))
#         g = int(np.interp(normalized_val, [0, 1], [int(middle_hex[3:5], 16), int(green_hex[3:5], 16)]))
#         b = int(np.interp(normalized_val, [0, 1], [int(middle_hex[5:7], 16), int(green_hex[5:7], 16)]))
    
#     # Convert interpolated RGB values to hex format for CSS color styling
#     hex_color = f'#{r:02x}{g:02x}{b:02x}'
    
#     return f'background-color: {hex_color}; color: black;'


# In[ ]:


import numpy as np

def custom_color_gradient(val, vmin=-100, vmax=100):
    green_hex = '#347c47'
    middle_hex = '#dcdcdc'
    lower_hex = '#b0343c'
    
    # Adjust the normalization to set the middle value as 0
    try:
        # Normalize the value to be between -1 and 1 with 0 as the midpoint
        normalized_val = (val - vmin) / (vmax - vmin) * 2 - 1
    except ZeroDivisionError:
        normalized_val = 0
    
    if normalized_val <= 0:
        # Interpolate between lower_hex and middle_hex for values <= 0
        r = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[1:3], 16), int(middle_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[3:5], 16), int(middle_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[5:7], 16), int(middle_hex[5:7], 16)]))
    else:
        # Interpolate between middle_hex and green_hex for values > 0
        r = int(np.interp(normalized_val, [0, 1], [int(middle_hex[1:3], 16), int(green_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [0, 1], [int(middle_hex[3:5], 16), int(green_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [0, 1], [int(middle_hex[5:7], 16), int(green_hex[5:7], 16)]))
    
    # Convert interpolated RGB values to hex format for CSS color styling
    hex_color = f'#{r:02x}{g:02x}{b:02x}'
    
    return f'background-color: {hex_color}; color: black;'


# In[53]:


def get_final_df(aspects_list,device):
    final_df = pd.DataFrame()
    device = device
    aspects_list = aspects_list

    # Iterate over each aspect and execute the query
    for aspect in aspects_list:
        # Construct the SQL query for the current aspect
        query = f"""
        SELECT Keywords,
               COUNT(CASE WHEN Sentiment = 'positive' THEN 1 END) AS Positive_Count,
               COUNT(CASE WHEN Sentiment = 'negative' THEN 1 END) AS Negative_Count,
               COUNT(CASE WHEN Sentiment = 'neutral' THEN 1 END) AS Neutral_Count,
               COUNT(*) as Total_Count
        FROM Sentiment_Data
        WHERE Aspect = '{aspect}' AND Product_Family LIKE '%{device}%'
        GROUP BY Keywords
        ORDER BY Total_Count DESC;
        """

        # Execute the query and get the result in 'key_df'
        key_df = ps.sqldf(query, globals())

        # Calculate percentages and keyword contribution
        total_aspect_count = key_df['Total_Count'].sum()
        key_df['Positive_Percentage'] = (key_df['Positive_Count'] / total_aspect_count) * 100
        key_df['Negative_Percentage'] = (key_df['Negative_Count'] / total_aspect_count) * 100
        key_df['Neutral_Percentage'] = (key_df['Neutral_Count'] / total_aspect_count) * 100
        key_df['Keyword_Contribution'] = (key_df['Total_Count'] / total_aspect_count) * 100

        # Drop the count columns
        key_df = key_df.drop(['Positive_Count', 'Negative_Count', 'Neutral_Count', 'Total_Count'], axis=1)

        # Add the current aspect to the DataFrame
        key_df['Aspect'] = aspect

        # Sort by 'Keyword_Contribution' and select the top 2 for the current aspect
        key_df = key_df.sort_values(by='Keyword_Contribution', ascending=False).head(2)

        # Append the results to the final DataFrame
        final_df = pd.concat([final_df, key_df], ignore_index=True)
        
    return final_df

# 'final_df' now contains the top 2 keywords for each of the top aspects


# In[13]:


# device_name = 'CoPilot for Mobile'
# device = device_name
# data = query_quant("Summarize the reviews of "+ device_name, [])
# total_reviews = data.loc[data['ASPECT'] == 'TOTAL', 'REVIEW_COUNT'].iloc[0]
# data['REVIEW_PERCENTAGE'] = data['REVIEW_COUNT'] / total_reviews * 100
# dataframe_as_dict = data.to_dict(orient='records')
# data_new = data
# data_new = data_new.dropna(subset=['ASPECT_SENTIMENT'])
# data_new = data_new[~data_new["ASPECT"].isin(["Generic", "Account", "Customer-Service", "Browser"])]
# vmin = data_new['ASPECT_SENTIMENT'].min()
# vmax = data_new['ASPECT_SENTIMENT'].max()
# styled_df = data_new.style.applymap(lambda x: custom_color_gradient(x, vmin, vmax), subset=['ASPECT_SENTIMENT'])
# data_filtered = data_new[data_new['ASPECT'] != 'TOTAL']
# data_sorted = data_filtered.sort_values(by='REVIEW_COUNT', ascending=False)
# top_four_aspects = data_sorted.head(4)
# aspects_list = top_four_aspects['ASPECT'].to_list()
# aspects_list
# formatted_aspects = ', '.join(f"'{aspect}'" for aspect in aspects_list)
# key_df = get_final_df(aspects_list, device)
# b =  key_df.to_dict(orient='records')
# print((query_detailed_summary("Summarize reviews of" + device + "for " +  formatted_aspects +  "Aspects which have following "+str(dataframe_as_dict)+ str(b) + "Reviews: ",[])))


# In[62]:


# device = device_name
# print(styled_df)
# selected_aspect = 'Connectivity'

# query = f"""
# SELECT Keywords,
#        COUNT(CASE WHEN Sentiment = 'positive' THEN 1 END) AS Positive_Count,
#        COUNT(CASE WHEN Sentiment = 'negative' THEN 1 END) AS Negative_Count,
#        COUNT(CASE WHEN Sentiment = 'neutral' THEN 1 END) AS Neutral_Count,
#        COUNT(*) as Total_Count
# FROM Sentiment_Data
# WHERE Aspect LIKE '%{selected_aspect}%' AND Product_Family LIKE '%{device}%'
# GROUP BY Keywords
# ORDER BY Total_Count DESC;
# """
# key_df = ps.sqldf(query, globals())
# total_aspect_count = key_df['Total_Count'].sum()
# key_df['Positive_Percentage'] = (key_df['Positive_Count'] / key_df['Total_Count']) * 100
# key_df['Negative_Percentage'] = (key_df['Negative_Count'] / key_df['Total_Count']) * 100
# key_df['Neutral_Percentage'] = (key_df['Neutral_Count'] / key_df['Total_Count']) * 100
# key_df['Keyword_Contribution'] = (key_df['Total_Count'] / total_aspect_count) * 100
# key_df = key_df.drop(['Positive_Count', 'Negative_Count', 'Neutral_Count', 'Total_Count'], axis=1)
# key_df = key_df.head(10)
# b =  key_df.to_dict(orient='records')
# print((query_detailed("Summarize reviews of" + device + "for " +  selected_aspect +  "Aspect which have following "+str(dataframe_as_dict)+ str(b) + "Reviews: ",[])))


# In[60]:


st.subheader("Product Consumer Review Summarization Tool")
device_name = st.text_input("Enter the Product name : ")
device = device_name
if device_name:
    data = query_quant("Summarize the reviews of "+ device_name, [])
    total_reviews = data.loc[data['ASPECT'] == 'TOTAL', 'REVIEW_COUNT'].iloc[0]
    data['REVIEW_PERCENTAGE'] = data['REVIEW_COUNT'] / total_reviews * 100
    dataframe_as_dict = data.to_dict(orient='records')
    data_new = data
    data_new = data_new.dropna(subset=['ASPECT_SENTIMENT'])
    data_new = data_new[~data_new["ASPECT"].isin(["Generic", "Account", "Customer-Service", "Browser"])]
    vmin = data_new['ASPECT_SENTIMENT'].min()
    vmax = data_new['ASPECT_SENTIMENT'].max()
    styled_df = data_new.style.applymap(lambda x: custom_color_gradient(x, vmin, vmax), subset=['ASPECT_SENTIMENT'])
    data_filtered = data_new[data_new['ASPECT'] != 'TOTAL']
    data_sorted = data_filtered.sort_values(by='REVIEW_COUNT', ascending=False)
    top_four_aspects = data_sorted.head(4)
    aspects_list = top_four_aspects['ASPECT'].to_list()
    formatted_aspects = ', '.join(f"'{aspect}'" for aspect in aspects_list)
    key_df = get_final_df(aspects_list, device)
    b =  key_df.to_dict(orient='records')
    st.write(query_detailed_summary("Summarize reviews of" + device + "for " +  formatted_aspects +  "Aspects which have following "+str(dataframe_as_dict)+ str(b) + "Reviews: ",[]))
    heat_map = st.checkbox("Would you like to see the Aspect wise sentiment of this Produt?")
    if heat_map:
        st.dataframe(styled_df)
        aspect_names = ['Microsoft Product', 'Interface', 'Code Generation', 'Image Generation', 'Productivity', 'Text Summarization/Generation', 'Connectivity', 'Compatibility', 'Privacy', 'Ease of Use', 'Reliability', 'Price', 'Innovation', 'Customization/Personalization', 'Generic']
        with st.form(key='my_form'):
            aspect_wise_sentiment = st.markdown("Select any one of the aspects to see what consumers reviews about that aspect..")
            selected_aspect = st.selectbox('Select an aspect to see consumer reviews:', aspect_names)
            submitted = st.form_submit_button('Submit')
            if submitted:

                query = f"""
                SELECT Keywords,
                       COUNT(CASE WHEN Sentiment = 'positive' THEN 1 END) AS Positive_Count,
                       COUNT(CASE WHEN Sentiment = 'negative' THEN 1 END) AS Negative_Count,
                       COUNT(CASE WHEN Sentiment = 'neutral' THEN 1 END) AS Neutral_Count,
                       COUNT(*) as Total_Count
                FROM Sentiment_Data
                WHERE Aspect LIKE '%{selected_aspect}%' AND Product_Family LIKE '%{device}%'
                GROUP BY Keywords
                ORDER BY Total_Count DESC;
                """
                key_df = ps.sqldf(query, globals())
                total_aspect_count = key_df['Total_Count'].sum()
                key_df['Positive_Percentage'] = (key_df['Positive_Count'] / key_df['Total_Count']) * 100
                key_df['Negative_Percentage'] = (key_df['Negative_Count'] / key_df['Total_Count']) * 100
                key_df['Neutral_Percentage'] = (key_df['Neutral_Count'] / key_df['Total_Count']) * 100
                key_df['Keyword_Contribution'] = (key_df['Total_Count'] / total_aspect_count) * 100
                key_df = key_df.drop(['Positive_Count', 'Negative_Count', 'Neutral_Count', 'Total_Count'], axis=1)
                key_df = key_df.head(10)
                b =  key_df.to_dict(orient='records')
                st.write(query_detailed("Summarize reviews of" + device + "for " +  selected_aspect +  "Aspect which have following "+str(dataframe_as_dict)+ str(b) + "Reviews: ",[]))
        
        
    
    


# In[ ]:


# device = device_name
# print(styled_df)
# selected_aspect = 'Performance'

# query = f"""
# SELECT Keywords,
#        COUNT(CASE WHEN Sentiment = 'Positive' THEN 1 END) AS Positive_Count,
#        COUNT(CASE WHEN Sentiment = 'Negative' THEN 1 END) AS Negative_Count,
#        COUNT(CASE WHEN Sentiment = 'Neutral' THEN 1 END) AS Neutral_Count,
#        COUNT(*) as Total_Count
# FROM Sentiment_Data
# WHERE Aspect LIKE '%{selected_aspect}%' AND Product_Family LIKE '%{device}%'
# GROUP BY Keywords
# ORDER BY Total_Count DESC;
# """
# key_df = ps.sqldf(query, globals())
# total_aspect_count = key_df['Total_Count'].sum()
# key_df['Positive_Percentage'] = (key_df['Positive_Count'] / key_df['Total_Count']) * 100
# key_df['Negative_Percentage'] = (key_df['Negative_Count'] / key_df['Total_Count']) * 100
# key_df['Neutral_Percentage'] = (key_df['Neutral_Count'] / key_df['Total_Count']) * 100
# key_df['Keyword_Contribution'] = (key_df['Total_Count'] / total_aspect_count) * 100
# key_df = key_df.drop(['Positive_Count', 'Negative_Count', 'Neutral_Count', 'Total_Count'], axis=1)
# key_df = key_df.head(10)
# b =  key_df.to_dict(orient='records')
# print((query_detailed("Summarize reviews of" + device + "for " +  selected_aspect +  "Aspect which have following "+str(dataframe_as_dict)+ str(b) + "Reviews: ",[])))


# In[54]:


# a = query_quant("What is the net sentiment of Github Copilot across different Geography",[])


# In[55]:


# a


# In[ ]:





# In[56]:


# dataframe_as_dict =  a.to_dict(orient='records')


# In[57]:


# dataframe_as_dict


# In[58]:


# print(query_detailed_summary("Summarize reviews of Github Copilot?. It has the following sentiment : " + str(dataframe_as_dict),[]))


# In[28]:


# device = 'Github Copilot'
# # print(styled_df)
# selected_aspect = 'Microsoft Product'

# query = f"""
# SELECT Keywords,
#        COUNT(CASE WHEN Sentiment = 'positive' THEN 1 END) AS Positive_Count,
#        COUNT(CASE WHEN Sentiment = 'negative' THEN 1 END) AS Negative_Count,
#        COUNT(CASE WHEN Sentiment = 'neutral' THEN 1 END) AS Neutral_Count,
#        COUNT(*) as Total_Count
# FROM Sentiment_Data
# WHERE Aspect LIKE '%{selected_aspect}%' AND Product_Family LIKE '%{device}%'
# GROUP BY Keywords
# ORDER BY Total_Count DESC;
# """
# key_df = ps.sqldf(query, globals())
# total_aspect_count = key_df['Total_Count'].sum()
# key_df['Positive_Percentage'] = (key_df['Positive_Count'] / key_df['Total_Count']) * 100
# key_df['Negative_Percentage'] = (key_df['Negative_Count'] / key_df['Total_Count']) * 100
# key_df['Neutral_Percentage'] = (key_df['Neutral_Count'] / key_df['Total_Count']) * 100
# key_df['Keyword_Contribution'] = (key_df['Total_Count'] / total_aspect_count) * 100
# key_df = key_df.drop(['Positive_Count', 'Negative_Count', 'Neutral_Count', 'Total_Count'], axis=1)
# key_df = key_df.head(10)
# b =  key_df.to_dict(orient='records')
# print((query_detailed("Summarize reviews of" + device + "for " +  selected_aspect +  "Aspect which have following "+str(dataframe_as_dict)+ str(b) + "Reviews: ",[])))


# In[ ]:




