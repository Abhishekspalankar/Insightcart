# Generated by Django 5.1.2 on 2024-11-04 14:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='AmazonProduct',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('price', models.CharField(max_length=50)),
                ('rating', models.CharField(blank=True, max_length=50, null=True)),
                ('link', models.URLField()),
                ('date_scraped', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='eBayProduct',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('price', models.CharField(max_length=50)),
                ('rating', models.CharField(blank=True, max_length=50, null=True)),
                ('link', models.URLField()),
                ('date_scraped', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='SnapdealProduct',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('price', models.CharField(max_length=50)),
                ('rating', models.CharField(blank=True, max_length=50, null=True)),
                ('link', models.URLField()),
                ('date_scraped', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
