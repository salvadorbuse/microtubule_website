---
layout: page
title: About
description: Information about the project, website, and links to the paper and SI
img: about.png # Add image post (optional)
caption: "A Serious Man (20??)  "
permalink: index.html
sidebar: true
---

---


# {{site.data.about.title}}
{{site.data.about.authors}}

{% for entry in site.data.about %}

{% if entry[0] != 'title' %}
{% if entry[0] != 'authors' %}
## {{entry[0]}}
{{entry[1]}}
{% endif %}
{% endfor %}

{% for fig in site.data.figure3 %}
<style>
.center {
  margin: auto;
  width: 60%;
  border: 3px solid #73AD21;
  padding: 10px;
}
</style>

<div class="center">
    <img src = "{{site.url}}/{{site.baseurl}}/assets/img/{{figure3.pic}}"> 
</div>
{% endif %}
{% endfor %}
