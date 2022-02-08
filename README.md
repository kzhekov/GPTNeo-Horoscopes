# Horoscope generation with GPT, Transformers, Fast API and GCP 
This project implements a complete cloud-hosted NLP pipeline for generating (hopefully) well-written horoscopes using Python 3 for the machine learning, data processing and API serving. It uses a variety of open source libraries such as Huggingface’s Transformers, PyTorch and FastAPI. It also has a small demo website implemented in HTML, CSS and JS that can be used to preview generated horoscopes for each of the zodiac signs. For a detailed explanation of the implementation, feel free to check out [this Medium article](https://medium.com/mlearning-ai/complete-horoscope-generation-pipeline-as-a-rest-api-using-gpt-neo-transformers-fast-api-and-gcp-1f3869cdf072).

This project consists in
- Scraping horoscopes from websites known to publish horoscopes daily, and store them as a CSV for later use or use other scraped horoscopes (only data is available)
- Using a pre-trained open-source version of GPT3 (GPTNeo) and fine-tune it to generate horoscopes when given a particular zodiac sign
- Creating a model wrapper that would handle data pre-processing, horoscope post-processing and deliver predictions from the horoscope model
- Creating a REST API using FastAPI that would handle prediction requests for the various zodiac signs
- Containerizing the API and the model using Docker and host it for public predictions on the Google Cloud Platform
- Creating a small website to test out the horoscope generation easily and scrape some rust off my web developing skills [that you can check out here](http://kamen.be/horoscope_codepen.html)

