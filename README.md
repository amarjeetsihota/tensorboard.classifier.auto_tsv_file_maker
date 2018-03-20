---


---

<h1 id="autogenerate-tsv-file-for-classifier">Autogenerate TSV file for classifier</h1>
<p>We have been developing AI solutions for a four months now and have built two commercial DNN Classifiers (for stockbroking clients)</p>
<p>One of the <strong>wow</strong> moments we get from clients is when we show the learnt embeddings for hiddenlayer_0 in tensorboard with labels.  We will perform a T-SNE in front of them and the clients can see the clusters of data that the classifier is using to make it’s decisions.</p>
<p>This code shows how we generate the TSV code generically for any dnn classifer.</p>
<h1 id="limitations">Limitations</h1>
<ul>
<li>Hash columns aren’t handled but would probably require an input<br>
customised class to be injected</li>
<li>Only labels for HiddenLayer0     are    generated</li>
</ul>
<h1 id="general-approach">General Approach</h1>
<p>Given most classifiers feature columns are one-hot category vectors, buckets and numeric columns, we have built an automatic tsv generator that</p>
<ol>
<li>Hooks into DNNClassifer using SessionRunHook</li>
<li>Grabs the    dnn/input_from_feature_columns/input_layer/concat:0<br>
tensor</li>
<li>For each    column type e.g. categorical, bucket, numeric, embedding<br>
a. Creates a    label based on categorical_colunn definition</li>
</ol>
<h2 id="data">Data</h2>
<p>The data used is from the “Mental Health in Tech Survey” dataset in Kaggle,  The train and test sets are in the data sub directory</p>
<h2 id="mhc.dnn.py"><a href="http://MHC.DNN.py">MHC.DNN.py</a></h2>
<p>Main code for training the classifier.  Standard code that uses the AgileDev_AI package for the heavy lifting.</p>
<h2 id="agile_dev_ai.dnn_helper">AGILE_DEV_AI.DNN_Helper</h2>
<p>You can rename the current file by clicking the file name in the navigation bar or by clicking the <strong>Rename</strong> button in the file explorer.</p>
<h2 id="section"></h2>
<pre><code></code></pre>

