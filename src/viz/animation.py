import ipywidgets as widgets
from bqplot import *
import qgrid
import numpy as np
import os

class BasicPlot(widgets.VBox):
    """
    base class that constructs a plot of a tracking frame
    """
    def __init__(self, 
                 X=[-57.8, 55], 
                 Y=[-39.5, 37.0],
                 width=506.7,
                 height=346.7, 
                 pitch_img='pitch_white.png', 
                 df_tracking=None, 
                 scaling=1.8, 
                 game_id=None):
        """
        init widget object
        """
        super().__init__()
        self.pitch_img = pitch_img
        self.game_id = game_id
        
        # Init plot
        self.image = self.__init_image(X,Y)
        self.team_scatter = self.__init_scatter(X, Y)
        self.ball_scatter = self.__init_scatter(X, Y, default_size=48)
        self.speed_scatter = self._BasicPlot__init_scatter(X, Y, default_size=15)
        self.line = self.__init_line(X, Y, default_size=10, close_path=False)

        self.default_marks = [
            self.image, 
            self.team_scatter, 
            self.ball_scatter,
            self.speed_scatter,
            self.line,
        ]

        self.fig = Figure(marks=self.default_marks, padding_x=0, padding_y=0, fig_margin={'top':0, 'bottom':0, 'left':30, 'right':30})
        self.fig.layout = widgets.Layout(width=f'{width*scaling}px', height=f'{height*scaling}px')
        self.children = [self.fig]

        # store tracking data
        self.df_tracking = df_tracking
        self.df_tracking['color'] = self.df_tracking.apply(self.mapTeamColour, axis=1)
        self.sample = None
        
        # add unique player identifier to each player
        self.__set_unique_jersey_no()
        
    
    def __init_image(self, X, Y):
        """
        init image upon which players are plotted
        """
        # read pitch image
        image_path = os.path.abspath(self.pitch_img)

        with open(image_path, 'rb') as f:
            raw_image = f.read()
            
        # set image as widget background
        ipyimage = widgets.Image(value=raw_image, format='png')

        scales_image = {'x': LinearScale(), 'y': LinearScale()}
        axes_options = {'x': {'visible': False}, 'y': {'visible': False}}

        image = Image(image=ipyimage, scales=scales_image, axes_options=axes_options)
        
        # Full screen
        image.x = X
        image.y = Y
        
        return image
        
    def __init_scatter(self, X, Y, default_size=64, selected_opacity=0.6, unselected_opacity=1.0):
        """
        init scatter plot that sets the players and the ball to the respective locations
        on the pitch
        """
        scales={'x': LinearScale(min=X[0], max=X[1]), 'y': LinearScale(min=Y[0], max=Y[1])}
        axes_options = {'x': {'visible': False}, 'y': {'visible': False}}
        
        
        team_scatter = Scatter(
                            scales= scales, 
                            default_size=default_size,
                            selected_style={'opacity': selected_opacity, 'stroke': 'Black'},
                            unselected_style={'opacity': unselected_opacity},
                            axes_options=axes_options)
        
        return team_scatter
    
    def __init_line(self, X, Y, close_path, default_size=64, selected_opacity=0.6, unselected_opacity=1.0):
        """
        init scatter plot that sets the players and the ball to the respective locations
        on the pitch
        """
        scales = {'x': LinearScale(min=X[0], max=X[1]), 'y': LinearScale(min=Y[0], max=Y[1])}
        axes_options = {'x': {'visible': False}, 'y': {'visible': False}}

        line = Lines(
            scales=scales,
            default_size=default_size,
            selected_style={'opacity': selected_opacity, 'stroke': 'Black'},
            unselected_style={'opacity': unselected_opacity},
            axes_options=axes_options)
        line.enable_move = False

        return line
    
    def __set_unique_jersey_no(self):
        """
        add unique jersey number to each player
        """
        # add column
        self.df_tracking.loc[:,'u_jersey_no'] = 0
        
        # convert jersey_no to int
        self.df_tracking['jersey_no'] = self.df_tracking['jersey_no'].astype(int)

        # fill column
        self.df_tracking.loc[self.df_tracking.team_id==1, 'u_jersey_no'] = self.df_tracking.loc[self.df_tracking.team_id==1, 'jersey_no'].astype(str)
        self.df_tracking.loc[self.df_tracking.team_id==2, 'u_jersey_no'] = self.df_tracking.loc[self.df_tracking.team_id==2, 'jersey_no'].astype(str) + ' '
        self.df_tracking.loc[self.df_tracking.team_id==4, 'u_jersey_no'] = ' '
    
    def set_data(self, selected_frame):
        self.sample = self.df_tracking.query(f'current_phase == {selected_frame[0]} and timeelapsed == {selected_frame[1]}')
        self.sample_teams = self.sample.query('team_id != 4')
        self.sample_ball = self.sample.query('team_id == 4')

        # Update team scatter
        self.team_scatter.x = self.sample_teams['pos_x']
        self.team_scatter.y = self.sample_teams['pos_y']


        self.team_scatter.names = self.sample_teams['u_jersey_no']
        self.team_scatter.colors=self.sample_teams['color'].values.tolist()

        # Update ball scatter
        self.ball_scatter.x = self.sample_ball['pos_x']
        self.ball_scatter.y = self.sample_ball['pos_y']
        self.ball_scatter.names = self.sample_ball['u_jersey_no']
        self.ball_scatter.colors=self.sample_ball['color'].values.tolist()

        # Update speed "arrows"
        self.line.x = np.concatenate((self.sample_teams['pos_x'].values.reshape((-1,1)),
                                      (self.sample_teams['pos_x'].values + self.sample_teams['speed_x'].values).reshape((-1,1))),
                                     axis=1)
        self.line.y = np.concatenate((self.sample_teams['pos_y'].values.reshape((-1,1)),
                                      (self.sample_teams['pos_y'].values + self.sample_teams['speed_y'].values).reshape((-1,1))),
                                     axis=1)
        self.line.colors = self.sample_teams['color'].values.tolist()  # Use apply with mapTeamColour

        # Update speed scatter
        self.speed_scatter.x = self.sample_teams['pos_x'].values + self.sample_teams['speed_x'].values
        self.speed_scatter.y = self.sample_teams['pos_y'].values + self.sample_teams['speed_y'].values
        self.speed_scatter.colors = self.sample_teams['color'].values.tolist()  # Use apply with mapTeamColour

    def mapTeamColour(self, row, pretty=False):
        if row['team_id'] == 1:
            if row['Goalkeeper'] == 'Yes':
                return 'lightblue'
            if row['shooter'] == True:
                return 'darkblue'
            else:
                return 'blue'
        elif row['team_id'] == 2:
            if row['Goalkeeper'] == 'Yes':
                return 'lightcoral' 
            if row['shooter'] == True:
                return 'darkred'
            else:
                return 'red'
        elif row['team_id'] == -1:
            return "green"
        elif row['team_id'] == 4:
            return "#ffb04f"
        else:
            return "#3c4766"

class InteractiveAnimation(widgets.VBox):
    """
    base class that constructs an interactive plot that allows moving around players/ball
    """
    def __init__(self, df_tracking):
        # store unique frame
        self.__frames = df_tracking[['current_phase','timeelapsed']].drop_duplicates().reset_index(drop=True)
        self.df_tracking = df_tracking.sort_values(by=['current_phase','timeelapsed','team_id', 'jersey_no'])
        
        
    def create_animation(self, 
                         X=[-57.8, 55], 
                         Y=[-39.5, 37.0],
                         width=506.7,
                         height=346.7, 
                         pitch_img='pitch_white.png', 
                         scaling=1.8,
                         step=1,
                         frame_rate=25):
        # create pitch widget
        self.animation_container = BasicPlot(
            X=X,
            Y=Y,
            width=width,
            height=height,
            pitch_img=pitch_img,
            scaling=scaling,
            df_tracking=self.df_tracking
        )
        
        # init animation widget
        self.animation_container.set_data((1,0))
        
        self.control_container = self.__add_to_layout(STEP=step,
                                                      frame_rate=frame_rate)
        
        return widgets.VBox([self.animation_container,
                             self.control_container])
        
    
    def __add_to_layout(self, frame_rate, STEP=1):
        """
        add slider elements to to widget container
        """
        # number of different frame
        no_frames = self.df_tracking.frame_count.nunique()
        
        # add play mode
        self.play = widgets.Play(interval=1000/frame_rate,
                                value=0,
                                step=STEP,
                                max=no_frames,
                                description="Press play",
                                disabled=False)
        
        # add slider
        self.slider = widgets.IntSlider(max=no_frames, continuous_update = False)
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))
        
        # add interactivity
        self.slider.observe(self.__update_data, names='value')
        self.slider.value
        
        # add checkbox for speed arrows
        self.show_speed_arrow = widgets.Checkbox(
            value=True,
            description='Show speed arrows',
            disabled=False
        )
        self.show_speed_arrow.observe(self.__show_arrow, names='value')
        
        return widgets.HBox(
            [self.play, 
             self.slider,
             self.show_speed_arrow,
            ]
        )
        
    def __update_data(self, change):
        """
        update pitch plot
        """
        self.animation_container.set_data(self.__get_frameset(change['new']))
        
        
    def __show_arrow(self, change):
        """
        update pitch plot
        """
        self.animation_container.speed_scatter.visible = change['new']
        self.animation_container.line.visible = change['new']
        
            
    def __get_frameset(self, row):
        """
        helper function to get respective currentphase and timeelapsed of new
        value on the slider object
        """
        frameset = tuple(self.__frames.iloc[row])
        return frameset
        
class InteractiveEventAnimation(widgets.VBox):
    """
    base class that constructs an interactive plot that allows moving around players/ball
    """
    def __init__(self, df_tracking, df_events, offset=0):
        """
        init widget object by calling parent class constructor
        input:
            positions_df: dataframe that contains tracking data
            events_df: dataframe that contains the event data for the match
            offset: number of frames that should be started before an event
        """
        # store unique frame
        self.__frames = df_tracking[['current_phase','timeelapsed']].drop_duplicates().reset_index(drop=True)
        self.df_tracking = df_tracking.sort_values(by=['current_phase','timeelapsed','team_id', 'jersey_no'])
        self.df_events = df_events
        self.offset = offset

        
        
    def create_event_animation(self, 
                         X=[-57.8, 55], 
                         Y=[-39.5, 37.0],
                         width=506.7,
                         height=346.7, 
                         pitch_img='pitch_white.png', 
                         scaling=1.8,
                         step=1,
                         frame_rate=10):
        """
        compose and return widget to be displayed
        input:
            X: x-coordinates pitch
            Y: y-coordinates pitch
            width: width (pixel) of image upon which is plotted
            height: height (pixel) of image upon which is plotted
            pitch_img: file that contains image of soccer pitch
            positions_df: dataframe that contains tracking data
            scaling: scale factor for pitch_img
            step: step parameter for widget "play"
            frame_rate: sampling frequency of data; needed to display the animation correctly
        """
        # create pitch widget
        self.__ani_obj = InteractiveAnimation(df_tracking=self.df_tracking)
        animation_container = self.__ani_obj.create_animation(X=X,
                                                              Y=Y,
                                                              width=width,
                                                              height=height,
                                                              pitch_img=pitch_img,
                                                              scaling=scaling,
                                                              step=step,
                                                              frame_rate=frame_rate)
        
        self.__add_to_layout()
        
        return widgets.VBox([animation_container,
                             self.__event_container])
        
    
    def __add_to_layout(self):
        """
        add slider elements to to widget container
        """
        # Define qgrid widget
        qgrid.set_grid_option('maxVisibleRows', 10)
        col_opts = { 
            # 'editable': False,
            'editable': True,

        }
           
        self.__event_container = qgrid.show_grid(self.df_events, show_toolbar=False, column_options=col_opts)
        self.__event_container.layout = widgets.Layout(width='920px')
           
        self.__event_container.observe(self.__on_row_selected, names=['_selected_rows'])
        

            
    def __on_row_selected(self, change):
        """
        callback for row selection: update selected points in scatter plot
        """

        # get selcted event
        filtered_df = self.__event_container.get_changed_df()
        event = filtered_df.iloc[change.new]

        # event = self.events_df.iloc[change.new]
        
        # find index to which slider needs to be set
        idx = self.__frames.query(f'current_phase=={event.current_phase.item()} and timeelapsed=={event.timeelapsed.item()}').index[0]
        
        # set slider
        self.__ani_obj.slider.value = int(idx - self.offset)