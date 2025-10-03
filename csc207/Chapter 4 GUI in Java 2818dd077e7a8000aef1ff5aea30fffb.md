# Chapter 4: GUI in Java

### 4.1 Creating and showing a window in Java

- Java Swing library: javax.swing
- Components: all visual objects

```java
SwingUtilities.invokeLater(() -> { // create and show window

JFrame frame = new JFrame("Intro JFrame Example"); // title of the window
frame.setMinimumSize(new java.awt.Dimension(300,200)); // minimu, size
frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // set quit program when clikc close
frame.pack(); // pack its content according to layout choice
frame.setVisible(true); // set visiblity

}
```

### 4.2 Java Swing Visual Components

- JFrame (is the whole window!)
- JPanel (like a div in HTML! Naturally horizontal, goes into content pane of JFrame)
- JLabel
- JTextField
- JButton

```java
    JPanel firstNamePanel = new JPanel();
    firstNamePanel.add(new JLabel("First Name:"));
    firstNamePanel.add(new JTextField(10));
```

```java
    JPanel mainPanel = new JPanel();
    mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS)); // To set layout to vertical
    mainPanel.add(firstNamePanel);
    mainPanel.add(lastNamePanel);
    mainPanel.add(buttonPanel);
```

### 4.3 Handling button clicks

- UI is event driven
- Any event (click, typing) creates an event object, and calls a method we specify. The method we write is the event listener.

```java
JButton submit = new JButton("Submit");

submit.addActionListneer(new ActionListener() {
		@Override
		public void actionPerformed(ActionEvent e){
				String firstName = firstNameField.getText();
        String lastName = lastNameField.getText();
        JOptionPane.showMessageDialog(null, "Hello " + firstName + " " + lastName);
    }
});

		}
}
```