#ifndef CANADARM_PLUGIN_HPP_
#define CANADARM_PLUGIN_HPP_

#include "rclcpp/rclcpp.hpp"
#include "gazebo/common/Plugin.hh"
#include "gazebo/common/common.hh"
#include "gazebo/physics/Model.hh"
#include "gazebo/physics/Link.hh"
#include "gazebo_msgs/msg/link_state.hpp"


namespace canadarm_gz_plugin
{
    class CANADARMPluginPrivate;
    class CANADARMPlugin : public gazebo::ModelPlugin
    {
    public:
        CANADARMPlugin();
        ~CANADARMPlugin(); 

        void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override;
        void Update();

    private:
        std::unique_ptr<CANADARMPluginPrivate> impl_;

        //node
        rclcpp::Node::SharedPtr node_;

        //plugin_parameter
        gazebo::physics::ModelPtr model_;
        gazebo::physics::LinkPtr base_link_;

        //publisher
        rclcpp::Publisher<gazebo_msgs::msg::LinkState>::SharedPtr base_link_publisher_;
        
        //message 
        gazebo_msgs::msg::LinkState base_state_;
    };
}
#endif 



