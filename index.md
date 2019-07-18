# Neuralcoreference Resolution

This work was done as a part of my summer internship at MT-NLP Lab, Language Technology Research Center, Kohli Center for Intelligent Systems, IIIT-Hyderabad.

The aim of this work was to explore neuralcoreference resolution on NPTEL-Lecture Transcripts provided by MHRD,Govt of India.

We applied State-the-art neuralcoreference models on the lecture transcripts, identified it's weak points and presented our solution to the problem.

In the following sections, we explain in detail, the background of this work, the problem associated and our solutions to it.

## Definitions (What is meant by coreference ?)

Coreference Resolution, is an extra-linguistic terminology. Coreferential terms
may have completely dierent “senses” and yet, by denition, they refer to the
same extra linguistic entity. Coreference treats entities in a way more similar
to how we understand discourse, i.e., by treating each entity as a unique entity in real time.

## Problem Statement

### NPTEL Introduction

NPTEL is an acronym for National Programme on Technology Enhanced Learning which is an initiative by seven Indian Institutes of Technology (IIT Bombay, Delhi, Guwahati, Kanpur, Kharagpur, Madras and Roorkee) and Indian Institute of Science (IISc) for creating course contents in engineering and science.

NPTEL as a project originated from many deliberations between IITs, Indian
Institutes of Management (IIMs) and Carnegie Mellon University (CMU) during the years 1999-2003. A proposal was jointly put forward by five IITs (Bombay, Delhi, Kanpur, Kharagpur and Madras) and IISc for creating contents for 100 courses as web based supplements and 100 complete video courses, for forty hours of duration per course. Web supplements were expected to cover materials that could be delivered in approximately forty hours. Five engineering branches (Civil, Computer Science, Electrical, Electronics and Communication and Mechanical) and core science programmes that all engineering students are required to take in their undergraduate engineering programme in India were chosen initially. Contents for the above courses were based on the model curriculum suggested by All India Council for Technical Education (AICTE) and the syllabi of major affiliating Universities in India.

### Issues

Since NPTEL transcripts are mostly available in English, it proves to be a
challenge when it is required by students whose native tongue might not be
english. To solve this chanllenge, we have english to hindi or indic language
systems. However these may fail to traslate some linguisitc entities because the entites are coreferent with others. To solve this challenge we perform coreference resolution on NPTEL Transcripts.

## Types of Errors

In this section we discuss the types of errors formed when performing coreference resolution.

* **Identification Error**
	This type of error occurs when a linguistic entity which is a candidate coreferent mention is not identified by the coreference algorithm.
	*For example, in the following sentence :* 
	
	*"**Bob** went to the supermarket and he bought a pair of socks."*
	
	The algorithm identifies only bob as a candidate coreferent mention and not the term  _**he**_.
	
* **False New**
	This type of error occurs when a candidate coreferent mention is classified as a new entity but in reality it is a part of an existing coreference chain.
	*For example, in the following sentence :* 
	
	*"**Bob** went to the supermarket and **he** bought a pair of socks."*
	
	The algorithm classifies _**he**_ as a part of a new coreference chain while it actually refers to _**Bob**_.
	
* **False Anaphoric**
	This type of error occurs when a potential coreferent mention is classified as being part of an existing coreference chain, whereas in reality, it is a new entity in itself.
	*For example, in the following sentence :* 
	
	*"**Bob** went to the supermarket and **he** bought a pair of socks. **Mary** also accompanied **him** to the store*"
	
	The algorithm classifies _**Mary**_ as a reference to _**Bob**_ whereas it is a new coreference chain.

* **Wrong Link**
	This type of error occurs when a pair of coreferent mentions belonging to two separate coreference chains are linked together.
*For example, in the following sentence :* 
	
	*"**Bob** went to the supermarket and **he** bought a pair of socks. **Mary** also accompanied **him** to the store. **She** bought **herself** a new dress.*"
	
	The algorithm classifies _**herself**_ or _**She**_ as a reference to _**Bob**_ instead of _**Mary**_.



## Inferences on the NPTEL Transcripts

On running the state of the art model on the NPTEL Transcript, we manually
evaluated the errors according to the definitions given above. We found that
the majority of the errors were either the first one or the second one. The
reason for the first type of error is the inability of the state-of-the-art model
to identify the relevant candidate mentions. This is because the training data
was composed of news articles which are different from lecture transcripts. The
reason for the second type of error is the hesitation of the model to form cluster
chains where the candidate mentions are very far apart. This is due to the fact
that during training, the candidate mentions are not more than few sentences
apart.However this is not the case with transcripts where the topic of the lecture might be introduced in the beginning and only referred back in the concluding
paragraph.

## Solutions

In the recent literature [memory models](https://arxiv.org/abs/1503.08895) have gained popularity for handling
long term dependencies and have been used extensively for tasks like question answering and dialog. We hypothesize that such models might aid in forming corefernece chains and will help reduce the second type of error which are caused due to longer discourses.



