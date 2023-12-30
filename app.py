import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px

# Load your data
# Load your data
portfolio_df = pd.read_csv('/archive/portfolio.csv')
profile_df = pd.read_csv('/archive/profile.csv')
transcript_df = pd.read_csv('/archive/transcript.csv')
df = pd.read_csv('/preprocessed_data.csv')


# Sidebar for navigation
st.sidebar.title('Navigation')
show_data_exploration = st.sidebar.checkbox('Data Exploration')
show_univariate_analysis = st.sidebar.checkbox('Univariate Analysis')
show_bivariate_analysis = st.sidebar.checkbox('Bivariate Analysis')
show_multivariate_analysis = st.sidebar.checkbox('Multivariate Analysis')
show_offer_analysis = st.sidebar.checkbox('Offer Analysis')
show_conclusion = st.sidebar.checkbox('Conclusion')

# Main app layout
st.title('Starbucks Capstone Challenge')

# Introduction
image_url = "https://miro.medium.com/max/1224/1*lv_IE5h2CnUEVUOmK_dgpg.jpeg"
st.image(image_url, caption="Starbucks Challenge", use_column_width=True)
st.write('## Introduction')
st.write('This project is my Capstone Challenge for Udacity’s Data Scientist Nanodegree. The project is in collaboration with Starbucks where we were given simulated data that mimics customer behavior on the Starbucks rewards app. The offer could be purely informational or it could include a discount such as BOGO (buy one get one free).')
st.write('')
st.write('From the data we received, it appears that Starbucks sent 10 different offers to its customers via a variety of different channels.')
st.write('')

st.write('## Datasets')
st.write('For this project, we received 3 datasets:')
st.write('')
st.write('1. Portfolio: dataset describing the characteristics of each offer type, including its offer type, difficulty, and duration.')
st.write('2. Profile: dataset containing information regarding customer demographics including age, gender, income, and the date they created an account for Starbucks Rewards.')
st.write('3. Transcript: dataset containing all the instances when a customer made a purchase, viewed an offer, received an offer, and completed an offer. It\'s important to note that if a customer completed an offer but never actually viewed the offer, then this does not count as a successful offer as the offer could not have changed the outcome.')
st.write('')

st.write('## Project Overview')
st.write('The purpose of this project is to complete an exploratory analysis to determine which demographic group responds best to which offer type. I will also create and compare different predictive models to evaluate which features contribute to a successful offer.')
st.write('')

st.write('## Performance Metrics')
st.write('The performance of each trained predictive model was measured using a test dataset. As this was a binary classification outcome, I used AUC, accuracy, f1 score, and confusion matrix as the performance metrics.')
#print seperate line 
st.write('---')

################################################

# Data Exploration Section
if show_data_exploration:
    st.header('Data Exploration')
    if st.checkbox('Show  Dataset'):
        st.write(df.head())

# Univariate Analysis Section
if show_univariate_analysis:
    st.write('## Univariate Analysis')

    st.subheader('Income by Gender Distribution')
    sns.displot(df, x="income", hue="gender", element="step", kde=True)

    membership_subs = df[df['year'] >= 2014].groupby(['year','month'], as_index=False).agg({'customer_id':'count'})
    st.subheader('Subsciptions by Month and Year')
    plt.figure(figsize=(10,8))
    sns.pointplot(x="month", y="customer_id", hue="year", data = membership_subs)
    plt.ylabel('Customer Subsciptions', fontsize = 12)
    plt.xlabel('Month', fontsize = 12)
    plt.title('Subsciptions by Month and Year')

    st.subheader('Distribution of The Number of Days a User Has Been a Member')
    # Display the histogram of 'became_member_on'
    plt.figure(figsize=(10, 6))
    plt.hist(profile_df['became_member_on'], bins=30, edgecolor='black')
    plt.title('Distribution of Membership Dates')
    plt.xlabel('Membership Date')
    plt.ylabel('Frequency')
    st.pyplot(plt)    
    plt.title('Distribution of The Number of Days a User Has Been a Member')

    # Count of different types of events
    event_counts = df[['offer completed', 'offer received', 'offer viewed', 'transaction']].sum()

    # Streamlit app
    st.title('Count of Different Types of Events')

    # Create a bar chart using Plotly
    fig = go.Figure(data=[go.Bar(x=event_counts.index, y=event_counts.values, marker_color='green')])

    # Set the title and axis labels
    fig.update_layout(xaxis_title='Event', yaxis_title='Count')

    # Show the plot using Streamlit's Plotly chart support
    st.plotly_chart(fig)


    # Streamlit app
    st.title('Number of Customers per Offer')
    # df['event'].value_counts()
    customers_per_offer = df.groupby('offer_id')['customer_id'].nunique()
    customers_per_offer=customers_per_offer.sort_values(ascending=False)
    print(customers_per_offer)


    # Create a bar chart using Plotly Express
    fig = px.bar(customers_per_offer, x=customers_per_offer.index, y=customers_per_offer.values, title='Number of Customers per Offer', color='customer_id')

    # Show the plot using Streamlit's Plotly chart support
    st.plotly_chart(fig)



    st.title('Number of Customers per Offer')

    # Create a bar chart using Plotly Express
    fig = px.bar(customers_per_offer, x=customers_per_offer.index, y=customers_per_offer.values, title='Number of Customers per Offer', color='customer_id')

    # Show the plot using Streamlit's Plotly chart support
    st.plotly_chart(fig)

    # Average transaction amount
    average_transaction = df['amount'].mean()

    # Display the average transaction amount
    st.write(f"Average Transaction Amount: {average_transaction}")

    # Count of offers by type
    offer_type_counts = df[['bogo', 'discount', 'informational']].sum()

    # Streamlit app for bar chart of offer type counts
    st.title('Count of Offers by Type')
    fig_bar = go.Figure(data=[go.Bar(x=offer_type_counts.index, y=offer_type_counts.values)])
    fig_bar.update_layout(xaxis_title='Offer Type', yaxis_title='Count')
    st.plotly_chart(fig_bar)

    # Streamlit app for pie chart of offer type distribution
    st.title('Distribution of Offer Types')
    fig_pie = px.pie(values=offer_type_counts.values, names=offer_type_counts.index)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie)

    # Streamlit app for distribution of offer event status per gender
    st.title('Distribution of Offer Event Status per Gender')
    sns_countplot_gender = sns.countplot(data=df, x='gender', hue='event')
    st.pyplot(sns_countplot_gender.figure)

    # Streamlit app for distribution of offer event status
    st.title('Distribution of Offer Event Status')
    sns_countplot_offer_event = sns.countplot(data=df[df['event']!='transaction'], x='offer_type', hue='event')
    st.pyplot(sns_countplot_offer_event.figure)

    # Streamlit app for distribution of offer event status per offer id
    st.title('Distribution of Offer Event Status per Offer ID')
    plt.figure(figsize=(12, 6))
    sns_countplot_offer_id = sns.countplot(data=df[df['event']!='transaction'], x='offer_id', hue='event')
    st.pyplot(sns_countplot_offer_id.figure)

    # Setting plot style
    sns.set(style="whitegrid")

    # Function to create histograms for numerical columns
    # Function to create histograms for numerical columns
    def plot_histogram(data, column, title, xlabel, bins=30):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[column], bins=bins, kde=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        return fig

    # Function to create count plots for categorical columns
    def plot_countplot(data, column, title, xlabel):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=column, data=data, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        return fig


    # Histograms for numerical columns
    st.pyplot(plot_histogram(df, 'time', 'Distribution of Time', 'Time'))
    st.pyplot(plot_histogram(df, 'amount', 'Distribution of Transaction Amounts', 'Amount'))
    st.pyplot(plot_histogram(df, 'age', 'Age Distribution', 'Age'))
    st.pyplot(plot_histogram(df, 'income', 'Income Distribution', 'Income'))

    # Count plots for categorical columns
    st.pyplot(plot_countplot(df, 'event', 'Event Type Distribution', 'Event Type'))
    st.pyplot(plot_countplot(df, 'gender', 'Gender Distribution', 'Gender'))
    #bivarate analysis
    # Streamlit app for correlation heatmap
    # Setting plot style
    sns.set(style="whitegrid")

    # Function to create histograms for numerical columns
    def plot_histogram(data, column, title, xlabel, bins=30):
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], bins=bins, kde=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        st.pyplot()

    # Function to create count plots for categorical columns
    def plot_countplot(data, column, title, xlabel):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=column, data=data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        st.pyplot()
################################################



# Bivariate Analysis Section
if show_bivariate_analysis:
    st.write('## Bivariate Analysis')
    st.title('Bivariate Analysis: Exploring Relationships Between Different Variables')

    # Function to create scatter plots for numerical vs numerical columns
    def plot_scatter(data, x_column, y_column, title, xlabel, ylabel):
        st.write(f"## {title}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=x_column, y=y_column, data=data, ax=ax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        st.pyplot(fig)

    # Function to create box plots for categorical vs numerical columns
    def plot_boxplot(data, x_column, y_column, title, xlabel, ylabel):
        st.write(f"## {title}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=x_column, y=y_column, data=data, ax=ax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        st.pyplot(fig)

    # Scatter plot: Age vs Income
    plot_scatter(df, 'age', 'income', 'Age vs Income', 'Age', 'Income')

    # Box plot: Event Type vs Amount
    plot_boxplot(df, 'event', 'amount', 'Event Type vs Transaction Amount', 'Event Type', 'Transaction Amount')

    # Box plot: Gender vs Income
    plot_boxplot(df, 'gender', 'income', 'Gender vs Income', 'Gender', 'Income')

    # Box plot: Offer Type (BOGO, Discount, Informational) vs Time
    plot_boxplot(df, 'bogo', 'time', 'BOGO Offer vs Time', 'BOGO Offer', 'Time')
    plot_boxplot(df, 'discount', 'time', 'Discount Offer vs Time', 'Discount Offer', 'Time')
    plot_boxplot(df, 'informational', 'time', 'Informational Offer vs Time', 'Informational Offer', 'Time')

    # Box plot: Offer Type (BOGO, Discount, Informational) vs Amount
    plot_boxplot(df, 'bogo', 'amount', 'BOGO Offer vs Transaction Amount', 'BOGO Offer', 'Transaction Amount')
    plot_boxplot(df, 'discount', 'amount', 'Discount Offer vs Transaction Amount', 'Discount Offer', 'Transaction Amount')
    plot_boxplot(df, 'informational', 'amount', 'Informational Offer vs Transaction Amount', 'Informational Offer', 'Transaction Amount')
    
# Multivariate Analysis
if show_multivariate_analysis:
    st.write('## Multivariate Analysis')
    # Streamlit app
    st.title('Multivariate Analysis: Exploring Relationships Among Multiple Variables')

    # Pair plot for selected numerical variables
    st.write("## Pair Plot of Age, Income, Amount, and Time")
    pair_plot = sns.pairplot(df[['age', 'income', 'amount', 'time']])
    pair_plot.fig.suptitle("Pair Plot of Age, Income, Amount, and Time", y=1.02)
    st.pyplot(pair_plot)

    # Heatmap for correlations among numerical variables
    st.write("## Correlation Matrix of Age, Income, Amount, and Time")
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[['age', 'income', 'amount', 'time']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Age, Income, Amount, and Time")
    st.pyplot(plt)

    # Creating a pivot table for average amount by gender and event
    pivot_table = df.pivot_table(values='amount', index='gender', columns='event', aggfunc='mean')

    # Heatmap for pivot table
    st.write("## Average Transaction Amount by Gender and Event Type")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='viridis')
    plt.title("Average Transaction Amount by Gender and Event Type")
    st.pyplot(plt)

################################################



if show_offer_analysis:
    st.write('## Offer Analysis')
    # Preparing data for time series analysis of offer response
    offer_response_time_corrected = df.groupby('time')[['offer received', 'offer viewed', 'offer completed']].sum().reset_index()

    # Time Series Analysis of Offer Response
    st.write("## Time Series Analysis of Offer Response")
    plt.figure(figsize=(15, 5))
    sns.lineplot(x='time', y='offer received', data=offer_response_time_corrected, label='Offer Received')
    sns.lineplot(x='time', y='offer viewed', data=offer_response_time_corrected, label='Offer Viewed')
    sns.lineplot(x='time', y='offer completed', data=offer_response_time_corrected, label='Offer Completed')
    plt.title('Time Series Analysis of Offer Response')
    plt.xlabel('Time')
    plt.ylabel('Number of Offers')
    plt.legend()
    st.pyplot(plt)

    # Creating KDE plots for income and age for both men and women
    st.write("## Age and Income Distribution for Men and Women")
    plt.figure(figsize=(15, 6))

    # KDE Plot for Age
    plt.subplot(1, 2, 1)
    sns.kdeplot(df[df['gender'] == 'M']['age'], label='Men', shade=True)
    sns.kdeplot(df[df['gender'] == 'F']['age'], label='Women', shade=True)
    plt.title('Age Distribution for Men and Women')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()

    # KDE Plot for Income
    plt.subplot(1, 2, 2)
    sns.kdeplot(df[df['gender'] == 'M']['income'], label='Men', shade=True)
    sns.kdeplot(df[df['gender'] == 'F']['income'], label='Women', shade=True)
    plt.title('Income Distribution for Men and Women')
    plt.xlabel('Income')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt)





    # Streamlit app
    st.title('Transformed Dataset Analysis')

    # Analysis 1: Distribution of Successful vs. Unsuccessful Offers
    profile_offer_distribution = df.groupby(['customer_id', 'offer_id']).agg({
        'offer received': 'sum',
        'offer viewed': 'sum',
        'offer completed': 'sum'
    }).reset_index()

    profile_offer_distribution['successful_offer'] = (
        (profile_offer_distribution['offer viewed'] >= 1) & (profile_offer_distribution['offer completed'] >= 1)
    )

    offer_outcome_counts = profile_offer_distribution['successful_offer'].value_counts().reset_index()
    offer_outcome_counts.columns = ['Offer Outcome', 'Count']

    st.write("## Distribution of Successful vs. Unsuccessful Offers")
    fig_bar = go.Figure(data=[go.Bar(x=offer_outcome_counts['Offer Outcome'], y=offer_outcome_counts['Count'])])
    fig_bar.update_layout(xaxis_title='Offer Outcome (Successful = True)', yaxis_title='Count')
    st.plotly_chart(fig_bar)

    # Analysis 2: Count for Offer Type and Count for Offer Type per Gender
    offer_type_counts = df[['bogo', 'discount', 'informational']].sum().reset_index()
    offer_type_counts.columns = ['Offer Type', 'Count']

    gender_offer_counts = df.groupby('gender')[['bogo', 'discount', 'informational']].sum().reset_index()

    st.write("## Count of Each Offer Type")
    fig_bar_offer_type = go.Figure(data=[go.Bar(x=offer_type_counts['Offer Type'], y=offer_type_counts['Count'])])
    fig_bar_offer_type.update_layout(xaxis_title='Offer Type', yaxis_title='Count')
    st.plotly_chart(fig_bar_offer_type)

    st.write("## Count of Offer Type per Gender")
    gender_offer_counts_melted = gender_offer_counts.melt(id_vars='gender', var_name='Offer Type', value_name='Count')
    fig_bar_gender_offer = go.Figure(data=[go.Bar(x=gender_offer_counts_melted['Offer Type'], y=gender_offer_counts_melted['Count'], 
                                                color=gender_offer_counts_melted['gender'], barmode='group')])
    fig_bar_gender_offer.update_layout(xaxis_title='Offer Type', yaxis_title='Count', legend_title='Gender')
    st.plotly_chart(fig_bar_gender_offer)

    # Analysis for count of events per offer_id
    offer_id_counts = df.groupby('offer_id')[['offer received', 'offer viewed', 'offer completed']].sum().reset_index()
    gender_offer_id_counts = df.groupby(['gender', 'offer_id'])[['offer received', 'offer viewed', 'offer completed']].sum().reset_index()

    st.write("## Count of Events per Offer ID")
    offer_id_counts_melted = offer_id_counts.melt(id_vars='offer_id', var_name='Offer Event', value_name='Count')
    fig_bar_offer_id = go.Figure(data=[go.Bar(x=offer_id_counts_melted['offer_id'], y=offer_id_counts_melted['Count'],
                                            color=offer_id_counts_melted['Offer Event'], barmode='group')])
    fig_bar_offer_id.update_layout(xaxis_title='Offer ID', yaxis_title='Count', legend_title='Offer Event')
    st.plotly_chart(fig_bar_offer_id)

    st.write("## Count of Events per Offer ID per Gender")
    gender_offer_id_counts_melted = gender_offer_id_counts.melt(id_vars=['gender', 'offer_id'], var_name='Offer Event', value_name='Count')
    fig_bar_gender_offer_id = go.Figure(data=[go.Bar(x=gender_offer_id_counts_melted['offer_id'], 
                                                    y=gender_offer_id_counts_melted['Count'],
                                                    color=gender_offer_id_counts_melted['Offer Event'],
                                                    facet_col=gender_offer_id_counts_melted['gender'],
                                                    facet_col_wrap=2)])
    fig_bar_gender_offer_id.update_layout(xaxis_title='Offer ID', yaxis_title='Count', legend_title='Offer Event')
    st.plotly_chart(fig_bar_gender_offer_id)

    # Analysis for Average Transaction and Total Expenses per Customer
    transaction_data = df[df['transaction'] == 1]
    total_expenses = transaction_data.groupby('customer_id')['amount'].sum().reset_index()
    transaction_counts = transaction_data.groupby('customer_id').size().reset_index(name='transaction_count')
    customer_expenses = total_expenses.merge(transaction_counts, on='customer_id')
    customer_expenses['average_transaction'] = customer_expenses['amount'] / customer_expenses['transaction_count']

    st.write("## Average Transaction per Customer and Total Expenses per Customer")
    fig_hist_avg_transaction = go.Figure(data=[go.Histogram(x=customer_expenses['average_transaction'], nbinsx=400)])
    fig_hist_total_expenses = go.Figure(data=[go.Histogram(x=customer_expenses['amount'], nbinsx=100)])

    fig_hist_avg_transaction.update_layout(title_text="Average Transaction per Customer", xaxis_title="Average Transaction ($)")
    fig_hist_total_expenses.update_layout(title_text="Total Expenses per Customer", xaxis_title="Total Expenses ($)")

    st.plotly_chart(fig_hist_avg_transaction)
    st.plotly_chart(fig_hist_total_expenses)

    st.subheader('Customer Age Distribution')
    fig, ax = plt.subplots()
    sns.histplot(profile_df['age'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader('Customer Income Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['income'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader('Age by Gender Distribution')
    sns.displot(profile_df, x="age", hue="gender", element="step", kde=True)
    st.pyplot()




    # Streamlit app
    st.title('Offer Analysis')

    # Histogram for Average Transaction per Customer and Total Expenses per Customer
    st.write("## Average Transaction per Customer and Total Expenses per Customer")

    # Assuming 'amount' column represents transaction amount and 'transaction' column to identify transactions
    transaction_data = df[df['transaction'] == 1]
    total_expenses = transaction_data.groupby('customer_id')['amount'].sum().reset_index()
    transaction_counts = transaction_data.groupby('customer_id').size().reset_index(name='transaction_count')
    customer_expenses = total_expenses.merge(transaction_counts, on='customer_id')
    customer_expenses['average_transaction'] = customer_expenses['amount'] / customer_expenses['transaction_count']

    # Plotting with Plotly
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Average Transaction per Customer", "Total Expenses per Customer"))

    # Average Transaction Plot
    fig.add_trace(go.Histogram(x=customer_expenses['average_transaction'], nbinsx=400), row=1, col=1)

    # Total Expenses Plot
    fig.add_trace(go.Histogram(x=customer_expenses['amount'], nbinsx=100), row=1, col=2)

    # Update layout
    fig.update_layout(height=400, width=800, showlegend=False)
    fig.update_xaxes(title_text="Average Transaction ($)", row=1, col=1)
    fig.update_xaxes(title_text="Total Expenses ($)", row=1, col=2)

    # Show the plot
    st.plotly_chart(fig)

    # Countplot for Successful Offers by Gender
    st.write("## Successful Offer by Gender")

    gender_dist = df.groupby(['gender'])['successful offer'].sum()

    fig = go.Figure(data=[go.Pie(labels=gender_dist.index, values=gender_dist)])
    fig.update_traces(marker=dict(colors=['coral', 'lightblue', 'green']),
                    hoverinfo='label+percent',
                    textinfo='label',
                    textfont=dict(size=12),
                    textposition='inside')
    fig.update_layout(title="Successful Offer by Gender")
    st.plotly_chart(fig)

    # Distribution Plots for Offer Duration and Offer Difficulty
    st.write("## Distribution of Offer Duration and Offer Difficulty")

    # Set the figure size before creating the plots
    fig, ax = plt.subplots(figsize=(10, 5))

    # Filter data for successful and unsuccessful offers
    success_data = df[df['successful offer'] == 1]['duration']
    unsuccess_data = df[df['successful offer'] == 0]['duration']

    # Plot the KDE for successful offers in green
    sns.distplot(success_data, hist=False, color='green', kde_kws={'shade': True})

    # Plot the KDE for unsuccessful offers in grey
    sns.distplot(unsuccess_data, hist=False, color='grey', kde_kws={'shade': True})

    # Add legend and title
    ax.legend(['Successful Offers', 'Unsuccessful Offers'], frameon=False)
    ax.set_title('Distribution of Offer Duration')

    # Hide the y-axis ticks and labels
    ax.get_yaxis().set_visible(False)

    # Show the plot
    st.pyplot(fig)

    # Set the figure size before creating the plots
    fig, ax = plt.subplots(figsize=(10, 5))

    # Filter data for successful and unsuccessful offers
    success_data = df[df['successful offer'] == 1]['difficulty']
    unsuccess_data = df[df['successful offer'] == 0]['difficulty']

    # Plot the KDE for successful offers in green
    sns.kdeplot(success_data, color='green', fill=True, label='Successful Offers')

    # Plot the KDE for unsuccessful offers in grey
    sns.kdeplot(unsuccess_data, color='grey', fill=True, label='Unsuccessful Offers')

    # Add legend and title
    ax.legend(frameon=False)
    ax.set_title('Distribution of Offer Difficulty')

    # Hide the y-axis ticks and labels
    ax.get_yaxis().set_visible(False)

    # Show the plot
    st.pyplot(fig)


    # Histogram for Average Transaction per Customer and Total Expenses per Customer
    st.write("## Average Transaction per Customer and Total Expenses per Customer")

    # Assuming 'amount' column represents transaction amount and 'transaction' column to identify transactions
    transaction_data = df[df['transaction'] == 1]
    total_expenses = transaction_data.groupby('customer_id')['amount'].sum().reset_index()
    transaction_counts = transaction_data.groupby('customer_id').size().reset_index(name='transaction_count')
    customer_expenses = total_expenses.merge(transaction_counts, on='customer_id')
    customer_expenses['average_transaction'] = customer_expenses['amount'] / customer_expenses['transaction_count']

    # Plotting with Plotly
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Average Transaction per Customer", "Total Expenses per Customer"))

    # Average Transaction Plot
    fig.add_trace(go.Histogram(x=customer_expenses['average_transaction'], nbinsx=400), row=1, col=1)

    # Total Expenses Plot
    fig.add_trace(go.Histogram(x=customer_expenses['amount'], nbinsx=100), row=1, col=2)

    # Update layout
    fig.update_layout(height=400, width=800, showlegend=False)
    fig.update_xaxes(title_text="Average Transaction ($)", row=1, col=1)
    fig.update_xaxes(title_text="Total Expenses ($)", row=1, col=2)

    # Show the plot
    st.plotly_chart(fig)

    # Countplot for Successful Offers by Gender
    st.write("## Successful Offer by Gender")

    gender_dist = df.groupby(['gender'])['successful offer'].sum()

    fig = go.Figure(data=[go.Pie(labels=gender_dist.index, values=gender_dist)])
    fig.update_traces(marker=dict(colors=['coral', 'lightblue', 'green']),
                    hoverinfo='label+percent',
                    textinfo='label',
                    textfont=dict(size=12),
                    textposition='inside')
    fig.update_layout(title="Successful Offer by Gender")
    st.plotly_chart(fig)

    # Distribution Plots for Offer Duration and Offer Difficulty
    st.write("## Distribution of Offer Duration and Offer Difficulty")

    # Set the figure size before creating the plots
    fig, ax = plt.subplots(figsize=(10, 5))

    # Filter data for successful and unsuccessful offers
    success_data = df[df['successful offer'] == 1]['duration']
    unsuccess_data = df[df['successful offer'] == 0]['duration']

    # Plot the KDE for successful offers in green
    sns.distplot(success_data, hist=False, color='green', kde_kws={'shade': True})

    # Plot the KDE for unsuccessful offers in grey
    sns.distplot(unsuccess_data, hist=False, color='grey', kde_kws={'shade': True})

    # Add legend and title
    ax.legend(['Successful Offers', 'Unsuccessful Offers'], frameon=False)
    ax.set_title('Distribution of Offer Duration')

    # Hide the y-axis ticks and labels
    ax.get_yaxis().set_visible(False)

    # Show the plot
    st.pyplot(fig)

    # Set the figure size before creating the plots
    fig, ax = plt.subplots(figsize=(10, 5))

    # Filter data for successful and unsuccessful offers
    success_data = df[df['successful offer'] == 1]['difficulty']
    unsuccess_data = df[df['successful offer'] == 0]['difficulty']

    # Plot the KDE for successful offers in green
    sns.kdeplot(success_data, color='green', fill=True, label='Successful Offers')

    # Plot the KDE for unsuccessful offers in grey
    sns.kdeplot(unsuccess_data, color='grey', fill=True, label='Unsuccessful Offers')

    # Add legend and title
    ax.legend(frameon=False)
    ax.set_title('Distribution of Offer Difficulty')

    # Hide the y-axis ticks and labels
    ax.get_yaxis().set_visible(False)

    # Show the plot
    st.pyplot(fig)

    # Scatter plot for Age vs. Income
    age_income_scatter = px.scatter(df, x='age', y='income', title='Age vs Income')
    st.plotly_chart(age_income_scatter)

    # Scatter plot for Amount vs. Time
    amount_time_scatter = px.scatter(df, x='time', y='amount', title='Amount vs Time')
    st.plotly_chart(amount_time_scatter)

    # Pie chart for Offer Type Distribution
    offer_type_pie = px.pie(df.offer_type.value_counts(), names=df.offer_type.value_counts().index, title='Distribution of Offer Types')
    st.plotly_chart(offer_type_pie)

    # Histogram for Age Distribution
    age_dist = px.histogram(df, x='age', title='Age Distribution')
    st.plotly_chart(age_dist)

    # Histogram for Income Distribution
    income_dist = px.histogram(df, x='income', title='Income Distribution')
    st.plotly_chart(income_dist)

    # Histogram for Transaction Amount Distribution
    amount_dist = px.histogram(df, x='amount', title='Transaction Amount Distribution')
    st.plotly_chart(amount_dist)

    # Heatmap for Correlation Matrix with one-hot encoded 'offer_id'
    st.write("## Correlation Matrix with One-Hot Encoded 'offer_id'")

    # One-hot encode the 'offer_id' column
    data_with_one_hot = pd.get_dummies(df, columns=['offer_id'])

    # Select relevant columns for the new heatmap, including one-hot encoded 'offer_id'
    relevant_columns = [
        'time', 'amount', 'offer completed', 'offer received', 'offer viewed',
        'transaction', 'difficulty', 'duration', 'reward', 'channel_email', 
        'channel_mobile', 'channel_social', 'channel_web', 'bogo', 'discount', 
        'informational'
    ]
    new_relevant_columns = [col for col in data_with_one_hot.columns if col.startswith('offer_id_')]

    new_relevant_columns.extend(relevant_columns)

    # Calculate the new correlation matrix with one-hot encoded 'offer_id'
    new_corr = data_with_one_hot[new_relevant_columns].corr()

    # Set up the matplotlib figure for the new heatmap
    plt.figure(figsize=(10, 10))

    # Draw the new heatmap
    sns.heatmap(new_corr, annot=False, cmap='coolwarm', cbar_kws={'shrink': .5})

    # Adjust the plot
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Show the heatmap
    st.pyplot(plt)



# Conclusion Section
if show_conclusion:
    st.header('Conclusion')
    st.write("""
    After building and evaluating different models, Random Forest, Logistic Regression, and SVC have shown the best performance and metrics for our data, with the highest accuracy. These performances could be further improved by tuning the parameters in future phases. It is important to avoid overfitting when building models. Additionally, it would be interesting to test other classification algorithms like SVM or K-NN and include them in the general comparison.

    ROC-AUC curve has been used to compare the different classification models:

    An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

    - True Positive Rate (TPR): Synonymous with recall and defined as TPR = TP / (TP + FN)
    - False Positive Rate (FPR): Defined as FPR = FP / (FP + TN)

    An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives.
    """)
