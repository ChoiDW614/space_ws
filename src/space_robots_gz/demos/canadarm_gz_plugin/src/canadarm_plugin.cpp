#include "canadarm_gz_plugin/canadarm_plugin.hpp"
#include "gazebo_ros/node.hpp"


namespace canadarm_gz_plugin
{
    class CANADARMPluginPrivate
    {
        public:

        CANADARMPluginPrivate() = default;
        virtual ~CANADARMPluginPrivate() = default;

        gazebo_ros::Node::SharedPtr nh_;
        gazebo::physics::ModelPtr model_;
        gazebo::event::ConnectionPtr update_connection_;

        bool start;
        bool stop_;
        std::thread thread_executor_spin_;
        rclcpp::executors::MultiThreadedExecutor::SharedPtr executor_;
        rclcpp::Duration control_period_ = rclcpp::Duration(1, 0);
        rclcpp::Time last_update_sim_time_ros_ = rclcpp::Time((int64_t)0, RCL_ROS_TIME);

        ignition::math::Pose3d base_pose_;
        ignition::math::v6::Vector3<double> base_linvel_;
        ignition::math::v6::Vector3<double> base_angvel_;
    };

    CANADARMPlugin::CANADARMPlugin()
    : impl_(std::make_unique<CANADARMPluginPrivate>())
    {
    }

    CANADARMPlugin::~CANADARMPlugin()
    {
        // Stop controller manager thread
        impl_->stop_ = true;
        impl_->executor_->cancel();
        impl_->thread_executor_spin_.join();

        // Disconnect from gazebo events
        impl_->update_connection_.reset();
    }

    void CANADARMPlugin::Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf)
    {
        impl_->model_ = model;
        impl_->nh_ = gazebo_ros::Node::Get(sdf);

        impl_->start = true;
        RCLCPP_INFO(impl_->nh_->get_logger(),"Starting Canadarm Plugin!");

        if (!impl_->model_) {
            RCLCPP_ERROR_STREAM(impl_->nh_->get_logger(), "Model is NULL");
            return;
        }

        if (!rclcpp::ok()) {
            RCLCPP_FATAL_STREAM(
            impl_->nh_->get_logger(),
            "A ROS node for Gazebo has not been initialized, unable to load plugin. " <<
            "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
            return;
        }
        RCLCPP_INFO(impl_->nh_->get_logger(), "Canadarm Plugin");

        std::string name;
        gazebo::physics::LinkPtr link;
        for(unsigned int i=0;i<impl_->model_->GetLinks().size();i++)
        {
            link = impl_->model_->GetLinks()[i];
            name = link->GetName();

            if(name.compare("ISS")==0)
            {
                this->base_link_ = link;
                break;
            }
        }
        RCLCPP_INFO(impl_ -> nh_->get_logger(),"Update Base Link");

        base_link_publisher_ = impl_->nh_->create_publisher<gazebo_msgs::msg::LinkState>("/base_link_state", 10);
        base_state_.link_name = "ISS";

        impl_->update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
                                        std::bind(&CANADARMPlugin::Update, this)
                                    );

        RCLCPP_INFO(impl_->nh_->get_logger(), "Loaded Canadarm Plugin.");
    }

    
    void CANADARMPlugin::Update()
    {
        // Update base link pose and twist
        impl_->base_pose_ = this->base_link_->WorldPose();
        impl_->base_linvel_ = this->base_link_->WorldLinearVel();
        impl_->base_angvel_ = this->base_link_->WorldAngularVel();

        // Update pose msg
        base_state_.pose.position.x = impl_->base_pose_.Pos().X();
        base_state_.pose.position.y = impl_->base_pose_.Pos().Y();
        base_state_.pose.position.z = impl_->base_pose_.Pos().Z();

        base_state_.pose.orientation.x = impl_->base_pose_.Rot().X();
        base_state_.pose.orientation.y = impl_->base_pose_.Rot().Y();
        base_state_.pose.orientation.z = impl_->base_pose_.Rot().Z();
        base_state_.pose.orientation.w = impl_->base_pose_.Rot().W();

        // Update twsit msg
        base_state_.twist.linear.x = impl_->base_linvel_.X();
        base_state_.twist.linear.y = impl_->base_linvel_.Y();
        base_state_.twist.linear.z = impl_->base_linvel_.Z();

        base_state_.twist.angular.x = impl_->base_angvel_.X();
        base_state_.twist.angular.y = impl_->base_angvel_.Y();
        base_state_.twist.angular.z = impl_->base_angvel_.Z();

        base_link_publisher_->publish(base_state_);
    }
    GZ_REGISTER_MODEL_PLUGIN(CANADARMPlugin)
}