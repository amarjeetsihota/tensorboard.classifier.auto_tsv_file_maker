---


---

<h1 id="autogenerate-tsv-file-for-a-dnn-classifier">Autogenerate TSV file for a DNN Classifier</h1>
<p>We have been developing AI solutions for a four months now and have built two commercial DNN Classifiers (for stockbroking clients)</p>
<p>One of the <strong>wow</strong> moments we get from clients is when we show the learnt embeddings for hiddenlayer_0 in tensorboard with labels.  We will perform a T-SNE in front of them and the clients can see the clusters of data that the classifier is using to make it’s decisions.</p>
<h2 id="objective">Objective</h2>
<p>Given most classifiers feature columns are one-hot category vectors, buckets and numeric columns, we have built an automatic tsv file generator that should be generic to all dnn classifer columns.</p>
<h1 id="limitations">Limitations</h1>
<ul>
<li>Hash columns aren’t handled in this codeset but would require an input<br>
customised class to be injected into the feature_column_helper</li>
<li>Only labels for HiddenLayer0     are    generated</li>
</ul>
<h2 id="data">Data</h2>
<p>The data used is from the “Mental Health in Tech Survey” dataset in Kaggle. Based on some input features,  the classifier tries to predicts the users response to “seek treatment” in the survey.</p>
<p>The train and test sets are in the data sub directory</p>
<h2 id="mhc.dnn.py"><a href="http://MHC.DNN.py">MHC.DNN.py</a></h2>
<p>This is the top level training routine and is standard implementation of a DNN Classifer with some specialized classes namely</p>
<pre><code>feature\_column\_helper = dnn_helpers.FeatureColumnLookup()  
</code></pre>
<p>tbhook = dnn_helpers.TBHookClass(feature_column_helper, LOG)</p>
<h2 id="agile_dev_ai.dnn_helper">AGILE_DEV_AI.DNN_Helper</h2>
<p>The guts of the code is here.</p>
<pre><code>General approach is:
 1. Hooks into DNNClassifer using SessionRunHook
 2. Grabs the    dnn/input_from_feature_columns/input_layer/concat:0
   tensor
 3. For each    column type e.g. categorical, bucket, numeric, embedding
	 a. Creates a    label based on categorical_colunn definition
</code></pre>
<h2 id="results">Results</h2>
<p>The classifier gives an accuracy of 76%.</p>

