{%- extends 'html_basic.tpl' -%}

{%- block header -%}
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="chrome=1">
    <title>Libstempo by vallis</title>

    <link rel="stylesheet" href="stylesheets/styles.css">
    <link rel="stylesheet" href="stylesheets/pygment_trac.css">
    <style>
        header {width: 250px;}
        section {width: 625px;}
        p {margin-bottom: 10px;}
        pre {margin-bottom: 10px;}
        .output pre {background: #f2f2f2; margin-bottom: 20px;}
        .prompt {display: none;}
        .anchor-link {display: none;}
    </style>
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
    <!--[if lt IE 9]>
    <script src="//html5shiv.googlecode.com/svn/trunk/html5.js"></script>
    <![endif]-->
    <script src="https://c328740.ssl.cf1.rackcdn.com/mathjax/latest/MathJax.js?config=TeX-AMS_HTML" type="text/javascript"></script>
    <script type="text/javascript">
        init_mathjax = function() {
            if (window.MathJax) {
                // MathJax loaded
                MathJax.Hub.Config({
                    tex2jax: {
                        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                        displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
                    },
                    displayAlign: 'left', // Change this to 'center' to center equations.
                    "HTML-CSS": {
                        styles: {'.MathJax_Display': {"margin": 0}}
                    }
                });
                MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
            }
        }
        init_mathjax();
    </script>
  </head>
{%- endblock header -%}

{% block body %}
<body>
    <div class="wrapper">
        <header>
            <h1>Libstempo</h1>
            <p>libstempo — a Python wrapper for tempo2</p>

            <p class="view"><a href="index.html">back to project homepage</a></p>
            <p class="view"><a href="https://github.com/vallis/libstempo">view on GitHub</a></p>
        </header>

        <section>
{{ super() }}
        </section>
{%- endblock body %}

{% block footer %}
        <footer>
            <p>this project is maintained by <a href="https://github.com/vallis">vallis</a></p>
            <p><small>an <a href="http://ipython.org/notebook.html">iPython notebook</a></br>
                      hosted on GitHub Pages<br/>
                      theme by <a href="https://github.com/orderedlist">orderedlist</a></small></p>
        </footer>
    </div>
</body>
</html>
{% endblock footer %}