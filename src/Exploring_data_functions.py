import matplotlib.pyplot as plt
import seaborn as sns

def violin_matrix(data,cols,target):
    fig,ax=plt.subplots(nrows=len(cols),ncols=1,figsize=(10,3*len(cols)))
    for i,col in enumerate(cols):
        sns.violinplot(data=data,x=target,y=col,ax=ax[i])
        ax[i].set_title(f'Violin plot of {col} vs {target}')
    plt.tight_layout()
    

def barplot_matrix(data, cols,target):
    figure,axs = plt.subplots(nrows=1, ncols=len(cols), figsize=(5 * len(cols),5))
    for col in cols:
        completion_rate = data.groupby(col, observed=False)[target].mean()
        completion_rate.plot(kind="bar", ax=axs[cols.index(col)])
        plt.ylabel("Booking completion rate")
        plt.xlabel(col)
        plt.title("Booking completion rate by " + col)
        plt.xticks(rotation=45)
        plt.tight_layout()

    